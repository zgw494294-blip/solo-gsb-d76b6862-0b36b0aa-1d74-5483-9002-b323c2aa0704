"""播放会话：单进程内存态，让已保存场景在播放中接纳后续修订。

会话在创建时复制场景当时的完整快照（字幕、分轨、changes、源版本），
游标从 0 毫秒开始，之后支持两种操作：

- 推进（advance）：提交非负整数 time_ms，游标只能不减；返回源版本、
  游标，以及该时刻处于活动区间 [start_ms, end_ms)（左闭右开）的字幕
  及其轨道。
- 事件推进（advance_events）：同样提交非负整数 time_ms 推进游标，
  额外返回“旧游标 < 事件时刻 <= time_ms”窗口内的全部进入（enter）/
  退出（exit）事件——一次推进跨过整条短字幕也不会漏掉。事件按时刻
  升序，同刻先退出后进入，同类按会话字幕顺序。0 毫秒已活动的字幕由
  创建响应的快照确定，不产生进入事件。事件只随本次推进生成，不保存
  历史、不提供重放。
- 临时隐藏说话人（set_hidden_speakers）：按 speaker 精确匹配隐藏字幕，
  仅作用于本会话的内存状态，默认空、不持久化。设置时游标不变，以当前
  游标的左闭右开活动区间判断显示差异：原可见而现隐藏的活动字幕发
  exit，原隐藏而现可见者发 enter；事件沿用原事件字段、时刻取当前游标，
  先 exit 后 enter，同类按会话字幕顺序，集合无变化则无事件。设置后，
  推进与事件推进的 active、events 均排除隐藏者，但游标照常推进；接纳
  不改变隐藏集合。完整快照（state()/finish）与分轨不受影响。
- 接纳（accept）：把场景当前快照并入会话。轨道数不得变；游标前已开始
  （start_ms <= 游标）的字幕，其 id、speaker、起止时间、locked_track
  必须不变，并固定原分轨；其余字幕可改、可删、可新增，但都必须在游标
  之后开始。随后按新快照顺序，以最少换轨与并列规则重算全部分轨。
  冲突或无解即失败且会话保持不变；成功则整体替换会话快照与源版本，
  场景本身不受影响。
- 结束（finish）：一次完整操作取回会话最终完整快照并把会话从注册表
  移除。结束与同一会话的推进、事件推进、隐藏切换、接纳在会话锁内串行
  生效；快照只取自轮到结束生效时的会话内存状态，不读取场景、因此不会
  纳入尚未接纳的场景修订。成功响应沿用创建会话的全部字段。结束后该会话
  ID 的一切请求（含已取得会话引用、在锁上等待的在途请求）均返回 404，
  并发结束最多一次成功。结束不改动源场景、其他会话或此前返回的快照，
  不保存结束记录，也不跨重启恢复。结束同时清除该会话的补取通道。
- 补取通道（channel）：每个会话可开启一个内存通道（重复开启 409），
  开启响应给出当前可见 active 与序号 0；此后普通推进、事件推进与隐藏
  切换产生的全部 enter/exit 按事件字段与排序入队，seq 从 1 连续编号，
  短字幕不漏记，接纳不重写已排队内容。补取提交 ack_seq（严格非负整数），
  只能确认此前补取已返回的最大 seq 以内的前缀（越界 422，旧确认不回退），
  返回确认位置与剩余未确认事件（按 seq 递增，重复补取内容与序号不变）。
  状态变更与入队在同一会话锁内一同生效，开启、补取与既有会话操作完整
  串行，失败不改会话或通道。通道仅内存态：不跨重启恢复、不做后台推送。
  进程内全部会话的通道共享一个未确认事件总量上限
  （CHANNEL_EVENT_LIMIT，正整数，默认 1024）：普通推进、事件推进与
  隐藏切换按本次实际生成的事件数在会话锁内先申请共享容量，再与状态变更、
  完整入队一同生效；容量不足整体失败（503），不推进游标、不改变隐藏
  集合、不消费 seq、不部分入队。未开启通道的操作不占容量，零事件操作在
  满额时仍可执行。容量仅在补取确认并删除前缀时按删除条数归还（读取与旧
  确认不归还）；结束会话归还该通道剩余队列且仅归还一次。

会话只存在于本进程内存中，不随重启保留。同一会话上的推进、事件推进、
隐藏切换、接纳、结束与补取通道的开启/补取通过会话级互斥锁串行化：
每次调用要么完整生效，要么完全无效。
接纳是一个完整操作——场景快照的读取（load_record 回调）、已开始字幕
校验、分轨重算与会话替换都在会话锁内完成，因此并发下迟到的接纳请求
读到的一定是轮到它生效时的场景快照，不会把会话回写到进入锁之前读到的
旧版本。结束在会话锁内置位 closed 墓碑并移除注册表条目：先取得会话
引用、在锁上排队的请求轮到它生效时会看到墓碑而失败（404），不会在
结束之后再对已结束的会话生效一次。
"""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .solver import Subtitle, solve
from .storage import SceneRecord


class AcceptConflict(Exception):
    """接纳场景当前快照失败（对应 HTTP 409）。

    原因包括：轨道数变化、已开始字幕被修改/删除、新增或修改的字幕
    未在游标之后开始、固定已开始字幕的原分轨后重算无解。
    """


class CursorRegression(Exception):
    """推进时间小于当前游标（对应 HTTP 422）。"""

    def __init__(self, cursor_ms: int, time_ms: int) -> None:
        self.cursor_ms = cursor_ms
        self.time_ms = time_ms
        super().__init__(
            f"time_ms={time_ms} 小于当前游标 cursor_ms={cursor_ms}，游标只能不减"
        )


class SceneUnavailable(Exception):
    """接纳生效时场景已不存在（对应 HTTP 404）。"""

    def __init__(self, scene_id: str) -> None:
        self.scene_id = scene_id
        super().__init__(f"场景 {scene_id!r} 不存在")


class SessionMissing(Exception):
    """注册表中不存在该会话 id（对应 HTTP 404）。"""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(f"播放会话 {session_id!r} 不存在")


class SessionFinished(Exception):
    """会话已结束：请求已取得引用但轮到它生效时墓碑已置位（对应 HTTP 404）。

    结束是完整操作：在会话锁内把 closed 置为 True 并移除注册表条目。
    此前已取得 PlaybackSession 引用、正在会话锁上排队的推进/事件推进/
    接纳/结束请求，轮到它生效时看到 closed 即按未知会话处理（404），
    不会在结束之后再对该会话生效一次。
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(f"播放会话 {session_id!r} 已结束")


class ChannelExists(Exception):
    """该会话已开启补取通道，重复开启（对应 HTTP 409）。"""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(f"播放会话 {session_id!r} 已开启补取通道")


class ChannelMissing(Exception):
    """该会话尚未开启补取通道（对应 HTTP 404）。"""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(f"播放会话 {session_id!r} 尚未开启补取通道")


class AckOutOfRange(Exception):
    """ack_seq 超过此前补取已返回的最大 seq（对应 HTTP 422，通道不变）。"""

    def __init__(self, ack_seq: int, max_returned_seq: int) -> None:
        self.ack_seq = ack_seq
        self.max_returned_seq = max_returned_seq
        super().__init__(
            f"ack_seq={ack_seq} 越界：此前补取已返回的最大 seq 为 "
            f"{max_returned_seq}"
        )


class ChannelCapacityExceeded(Exception):
    """本次操作实际产生的事件数无法获得共享容量（对应 HTTP 503，整体失败）。

    补取通道的未确认事件总量按进程内全部会话共享：申请量使在用量超过
    CHANNEL_EVENT_LIMIT 时抛出。失败操作不推进游标、不改变隐藏集合、
    不消费 seq、不部分入队、不占用任何容量；容量经确认删除或结束释放后
    原请求可原样重试。
    """

    def __init__(self, requested: int, limit: int, in_use: int) -> None:
        self.requested = requested
        self.limit = limit
        self.in_use = in_use
        super().__init__(
            f"补取通道未确认事件总量已满：本次申请 {requested} 条，"
            f"上限 {limit} 条，当前已用 {in_use} 条"
        )


# 进程内全部补取通道共享的未确认事件总量默认上限；可用环境变量
# CHANNEL_EVENT_LIMIT 覆盖（正整数字符串），非法配置拒绝启动。
DEFAULT_CHANNEL_EVENT_LIMIT = 1024


def channel_event_limit_from_env(
    raw: Optional[str], *, default: int = DEFAULT_CHANNEL_EVENT_LIMIT
) -> int:
    """解析 CHANNEL_EVENT_LIMIT：未设置取默认值；显式空串/空白视为非法。

    仅接受纯十进制正整数（"1024"）：空串、0、负数、小数、科学计数法、
    布尔值、含空白或其他字符的写法一律抛 ValueError——启动阶段据此拒绝
    启动。Python 对超大整数原生支持，不设人为上限。
    """
    if raw is None:
        return default
    if not raw or not raw.isascii() or not raw.isdigit() or int(raw) <= 0:
        raise ValueError(
            "CHANNEL_EVENT_LIMIT 必须为正整数（十进制，不含空白或其他字符），"
            f"非法配置：{raw!r}"
        )
    return int(raw)


class ChannelCapacity:
    """进程内全部补取通道共享的未确认事件总量（内存态，不持久化）。

    in_use 为当前所有已开启通道中“已入队且未被确认删除”的事件总数。
    生产方在会话锁内先 reserve 本次实际生成的事件数，再做状态变更与完整
    入队——reserve 不足即整体失败，不留下任何占用；确认删除前缀与结束
    通道时按实际条数 release。容量锁只在会话锁内短暂持有，从不反向等待
    会话锁或注册表锁，因此与确认、结束、生产并发时不会死锁。
    """

    def __init__(self, limit: int) -> None:
        if limit <= 0:
            raise ValueError("通道事件容量上限必须为正整数")
        self.limit = limit
        self.in_use = 0
        self._lock = threading.Lock()

    def reserve(self, count: int) -> None:
        """申请 count 个事件容量；不足抛 ChannelCapacityExceeded（不占用）。"""
        if count <= 0:
            return
        with self._lock:
            if self.in_use + count > self.limit:
                raise ChannelCapacityExceeded(count, self.limit, self.in_use)
            self.in_use += count

    def release(self, count: int) -> None:
        """归还 count 个事件容量（确认删除前缀 / 结束通道时调用）。"""
        if count <= 0:
            return
        with self._lock:
            self.in_use -= count
            # 容量恒等于各通道现存未确认事件之和，不得泄漏成负数
            if self.in_use < 0:
                self.in_use = 0
                raise RuntimeError("补取通道共享容量被超额释放")


@dataclass
class ChannelState:
    """补取通道的内存状态（仅属于会话，不持久化、不跨重启保留）。

    events 为尚未确认的已排队事件：每项含 seq 与事件推进的事件字段
    （id/speaker/track/time_ms/type），按 seq 递增；next_seq 为下一个
    事件序号（从 1 起连续编号）；acked_seq 为确认位置（已确认的最大
    序号，只增不回退）；max_returned_seq 为此前补取响应实际返回过的
    最大 seq——ack_seq 不得超过它，越界按 422 拒绝。
    """

    events: List[Dict[str, Any]] = field(default_factory=list)
    next_seq: int = 1
    acked_seq: int = 0
    max_returned_seq: int = 0


@dataclass
class PlaybackSession:
    """一次播放会话的内存状态；subtitles 与 tracks 按下标对齐。"""

    session_id: str
    scene_id: str
    source_revision: int
    track_count: int
    subtitles: List[Dict[str, Any]]
    tracks: List[int]
    changes: int
    cursor_ms: int = 0
    # 临时隐藏的说话人集合：按 speaker 精确匹配，仅影响本会话的
    # active/events 视图；默认空，接纳后保留，不随会话持久化。
    hidden_speakers: set = field(default_factory=set)
    # 补取通道：开启后记录普通推进/事件推进/隐藏切换产生的全部
    # enter/exit；仅内存态，未开启为 None，结束会话时清除。
    channel: Optional[ChannelState] = None
    # 结束墓碑：finish 在会话锁内置为 True 并移除注册表条目。此后即使
    # 仍持有本对象引用的在途请求，在锁内读到 closed 也一律按 404 失败。
    closed: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def active(self) -> List[Dict[str, Any]]:
        """游标时刻处于 [start_ms, end_ms) 内且未隐藏的字幕及轨道。

        隐藏只过滤播放视图：区间判断仍为左闭右开，分轨与完整快照不变。
        """
        hidden = self.hidden_speakers
        return [
            {**sub, "track": track}
            for sub, track in zip(self.subtitles, self.tracks)
            if sub["speaker"] not in hidden
            and sub["start_ms"] <= self.cursor_ms < sub["end_ms"]
        ]

    def state(self) -> Dict[str, Any]:
        """会话完整快照（纯数据，可直接作为响应体）。"""
        return {
            "session_id": self.session_id,
            "scene_id": self.scene_id,
            "source_revision": self.source_revision,
            "cursor_ms": self.cursor_ms,
            "track_count": self.track_count,
            "changes": self.changes,
            "subtitles": [dict(s) for s in self.subtitles],
            "assignments": [
                {"id": s["id"], "track": t}
                for s, t in zip(self.subtitles, self.tracks)
            ],
        }


def _identity(sub: Dict[str, Any]) -> Tuple[Any, ...]:
    """已开始字幕在接纳时必须保持不变的字段组合。"""
    return (
        sub["id"],
        sub["speaker"],
        sub["start_ms"],
        sub["end_ms"],
        sub.get("locked_track"),
    )


def _window_events(
    session: PlaybackSession, old_cursor: int, time_ms: int
) -> List[Dict[str, Any]]:
    """（旧游标, time_ms] 窗口内的全部进入/退出事件（排除隐藏说话人）。

    字幕在 start_ms 进入、end_ms 退出（区间左闭右开）；一次采样间隔内
    开始又结束的短字幕会同时给出进入与退出，不会漏。排序与事件推进一致：
    时刻升序，同刻先退出后进入，同类按会话字幕顺序。
    """
    # 每项为 (时刻, 同刻次序, 会话顺序, 事件体)；同刻次序令 exit 在前
    pending: List[Tuple[int, int, int, Dict[str, Any]]] = []
    hidden = session.hidden_speakers
    for order, (sub, track) in enumerate(
        zip(session.subtitles, session.tracks)
    ):
        # 隐藏说话人的字幕既不进入 active，也不产生进入/退出事件
        if sub["speaker"] in hidden:
            continue
        start_ms = sub["start_ms"]
        end_ms = sub["end_ms"]
        base = {
            "id": sub["id"],
            "speaker": sub["speaker"],
            "track": track,
        }
        if old_cursor < start_ms <= time_ms:
            pending.append(
                (start_ms, 1, order, {**base, "time_ms": start_ms, "type": "enter"})
            )
        if old_cursor < end_ms <= time_ms:
            pending.append(
                (end_ms, 0, order, {**base, "time_ms": end_ms, "type": "exit"})
            )
    pending.sort(key=lambda item: (item[0], item[1], item[2]))
    return [item[3] for item in pending]


def _channel_enqueue(
    session: PlaybackSession, events: List[Dict[str, Any]]
) -> None:
    """把本次操作产生的事件按序追加到补取通道（未开启则为无操作）。

    与产生事件的状态变更（游标推进、隐藏集合替换）在同一会话锁内一同
    生效：seq 从 1 起连续分配；已排队内容自此不被改写（接纳场景修订
    也不影响）。调用方须已先通过 ChannelCapacity.reserve 按本批事件数
    申请到共享容量（未开启通道时无需申请），并已持有会话锁。
    """
    channel = session.channel
    if channel is None:
        return
    for event in events:
        channel.events.append({"seq": channel.next_seq, **event})
        channel.next_seq += 1


class PlaybackManager:
    """会话注册表：单进程内存实现，不持久化、不跨重启保留。

    锁序约定为“会话锁 -> 注册表锁”：finish 先在注册表中查到会话，再
    进入会话锁，最后在会话锁内取注册表锁移除条目。其余路径从不持有
    注册表锁等待会话锁，因此不存在反向等待、不会死锁。
    """

    def __init__(
        self,
        channel_event_limit: int = DEFAULT_CHANNEL_EVENT_LIMIT,
    ) -> None:
        self._sessions: Dict[str, PlaybackSession] = {}
        self._registry_lock = threading.Lock()
        # 进程内全部会话的通道共享同一未确认事件总量
        self._channel_capacity = ChannelCapacity(channel_event_limit)

    def create(self, record: SceneRecord) -> Dict[str, Any]:
        """按场景当前快照创建会话：复制完整快照，游标从 0 毫秒开始。"""
        subtitles = [dict(s) for s in record.subtitles]
        track_by_id = {a["id"]: a["track"] for a in record.assignments}
        session = PlaybackSession(
            session_id=uuid.uuid4().hex,
            scene_id=record.scene_id,
            source_revision=record.revision,
            track_count=record.track_count,
            subtitles=subtitles,
            tracks=[track_by_id[s["id"]] for s in subtitles],
            changes=record.changes,
        )
        state = session.state()
        with self._registry_lock:
            self._sessions[session.session_id] = session
        return state

    def get(self, session_id: str) -> Optional[PlaybackSession]:
        with self._registry_lock:
            return self._sessions.get(session_id)

    @property
    def channel_event_limit(self) -> int:
        return self._channel_capacity.limit

    @property
    def channel_events_in_use(self) -> int:
        return self._channel_capacity.in_use

    def advance(self, session: PlaybackSession, time_ms: int) -> Dict[str, Any]:
        """推进游标（只能不减），返回源版本、游标与活动字幕及轨道。

        响应保持原样（不含事件）；若已开启补取通道，本次推进跨过的全部
        进入/退出事件（与事件推进同一窗口与排序，短字幕不漏记）先按实际
        条数申请进程内共享容量，再与游标更新、完整入队在同一会话锁内一同
        生效。容量不足抛 ChannelCapacityExceeded（503）：游标不推进、
        seq 不消费、不部分入队；未开启通道或零事件不占容量，满额时仍可
        推进。
        """
        with session.lock:
            if session.closed:
                raise SessionFinished(session.session_id)
            if time_ms < session.cursor_ms:
                raise CursorRegression(session.cursor_ms, time_ms)
            old_cursor = session.cursor_ms
            # 普通推进的响应不含事件，但补取通道要完整记录跨过的 enter/exit
            pending = (
                _window_events(session, old_cursor, time_ms)
                if session.channel is not None
                else []
            )
            # 容量先于状态变更申请：不足则整体失败，会话与通道均不变
            self._channel_capacity.reserve(len(pending))
            session.cursor_ms = time_ms
            _channel_enqueue(session, pending)
            return {
                "session_id": session.session_id,
                "scene_id": session.scene_id,
                "source_revision": session.source_revision,
                "cursor_ms": session.cursor_ms,
                "active": session.active(),
            }

    def advance_events(
        self, session: PlaybackSession, time_ms: int
    ) -> Dict[str, Any]:
        """推进游标并返回本次跨过的全部进入/退出事件。

        事件窗口为（旧游标, time_ms]：旧游标 < 事件时刻 <= time_ms。
        字幕在 start_ms 进入、end_ms 退出（区间左闭右开）；因此一次
        采样间隔内开始又结束的短字幕会同时给出退出与进入事件，不会漏。
        0 毫秒已活动（start_ms == 0）的字幕由创建会话时的快照确定，
        初始游标即 0，不产生进入事件。事件按时刻升序；同刻先退出后
        进入；同类按会话字幕顺序。time_ms 等于当前游标时事件为空。

        事件生成、游标更新与活动字幕计算基于同一份会话快照，在会话锁
        内一次完成，与推进、接纳、结束完整串行。倒退抛 CursorRegression，
        会话保持不变。事件不保存历史、不提供重放。隐藏说话人的字幕不
        进入 active，也不产生事件；游标仍照常推进。若已开启补取通道，
        本次事件先按实际条数申请进程内共享容量，再与游标更新、完整入队
        一同生效；容量不足抛 ChannelCapacityExceeded（503）：游标不推进、
        seq 不消费、不部分入队。未开启通道不占容量；零事件推进在满额时
        仍可执行。
        """
        with session.lock:
            if session.closed:
                raise SessionFinished(session.session_id)
            if time_ms < session.cursor_ms:
                raise CursorRegression(session.cursor_ms, time_ms)
            old_cursor = session.cursor_ms
            events = _window_events(session, old_cursor, time_ms)
            # 仅已开启通道时占用共享容量：申请先于游标更新与入队，
            # 不足则本次推进整体失败（响应事件同样不返回）
            if session.channel is not None:
                self._channel_capacity.reserve(len(events))
            session.cursor_ms = time_ms
            _channel_enqueue(session, events)
            return {
                "session_id": session.session_id,
                "scene_id": session.scene_id,
                "source_revision": session.source_revision,
                "cursor_ms": session.cursor_ms,
                "events": events,
                "active": session.active(),
            }

    def set_hidden_speakers(
        self, session: PlaybackSession, hidden_speakers: List[str]
    ) -> Dict[str, Any]:
        """替换本会话临时隐藏的说话人集合，并返回当前游标下的显示变更。

        按 speaker 精确匹配；hidden_speakers 由调用方保证为无重复、非空
        字符串序列（非法集合在 HTTP 层按 422 拒绝，本方法不负责校验），
        允许包含会话字幕中尚未出现（甚至永不出现）的说话人。游标不移动：
        以当前游标的左闭右开活动区间 [start_ms, end_ms) 判断新旧可见性——

        - 旧集合可见、新集合隐藏且此刻活动的字幕，产生 exit；
        - 旧集合隐藏、新集合可见且此刻活动的字幕，产生 enter；
        - 事件沿用事件推进的字段（id/speaker/track/time_ms/type），
          time_ms 取当前游标；先全部 exit 后全部 enter，同类按会话字幕
          顺序；新旧集合下显示状态相同的字幕不产生事件，集合无变化则
          事件为空。

        设置成功后，推进与事件推进的 active、events 都排除隐藏者，但
        游标照常推进；完整快照（state()/finish）、分轨与冻结规则不变。
        设置仅存于会话内存、不持久化，接纳不改变隐藏集合。与推进、事件
        推进、接纳、结束在会话锁内完整串行；会话已结束抛 SessionFinished。
        已开启通道时，显示变更事件先按实际条数申请进程内共享容量，再与
        隐藏集合替换、完整入队一同生效；容量不足抛 ChannelCapacityExceeded
        （503）：隐藏集合不变、seq 不消费、不部分入队。未开启通道不占
        容量；集合无变化（零事件）在满额时仍可切换。
        """
        with session.lock:
            if session.closed:
                raise SessionFinished(session.session_id)
            old_hidden = session.hidden_speakers
            new_hidden = set(hidden_speakers)
            cursor = session.cursor_ms
            exits: List[Tuple[int, Dict[str, Any]]] = []
            enters: List[Tuple[int, Dict[str, Any]]] = []
            for order, (sub, track) in enumerate(
                zip(session.subtitles, session.tracks)
            ):
                speaker = sub["speaker"]
                was_hidden = speaker in old_hidden
                is_hidden = speaker in new_hidden
                if was_hidden == is_hidden:
                    continue
                # 仅对当前游标左闭右开区间内的活动字幕发显示变更事件
                if not (sub["start_ms"] <= cursor < sub["end_ms"]):
                    continue
                event = {
                    "id": sub["id"],
                    "speaker": speaker,
                    "track": track,
                    "time_ms": cursor,
                    "type": "exit" if is_hidden else "enter",
                }
                (exits if is_hidden else enters).append((order, event))
            exits.sort(key=lambda item: item[0])
            enters.sort(key=lambda item: item[0])
            # 先 exit 后 enter；同类已按会话字幕顺序
            events = [item[1] for item in exits] + [
                item[1] for item in enters
            ]
            # 容量先于隐藏集合替换申请：不足则集合不变、入队不发生
            if session.channel is not None:
                self._channel_capacity.reserve(len(events))
            session.hidden_speakers = new_hidden
            # 与隐藏集合的替换一同生效：显示变更事件同步入队（未开启则无操作）
            _channel_enqueue(session, events)
            return {
                "session_id": session.session_id,
                "scene_id": session.scene_id,
                "source_revision": session.source_revision,
                "cursor_ms": cursor,
                # 回显本次接纳的集合：去重后保持提交顺序
                "hidden_speakers": list(hidden_speakers),
                "events": events,
                "active": session.active(),
            }

    def open_channel(self, session: PlaybackSession) -> Dict[str, Any]:
        """为会话开启补取通道：返回当前可见活动字幕与起始序号 0。

        每个会话最多一个通道，重复开启抛 ChannelExists（409，已有通道
        不变）。开启只建立空队列，不补记开启之前的事件——客户端以本响应
        的 active 作为初始可见集合，此后的 enter/exit 由通道记录。与
        既有会话操作在会话锁内完整串行；会话已结束抛 SessionFinished。
        """
        with session.lock:
            if session.closed:
                raise SessionFinished(session.session_id)
            if session.channel is not None:
                raise ChannelExists(session.session_id)
            session.channel = ChannelState()
            return {
                "session_id": session.session_id,
                "scene_id": session.scene_id,
                "source_revision": session.source_revision,
                "cursor_ms": session.cursor_ms,
                "seq": 0,
                "active": session.active(),
            }

    def fetch_channel(
        self, session: PlaybackSession, ack_seq: int
    ) -> Dict[str, Any]:
        """补取未确认事件并确认已收到的事件前缀。

        ack_seq 只能确认此前补取已返回的最大 seq 以内的前缀：越界抛
        AckOutOfRange（422，通道不变）；小于等于已确认位置的旧确认不
        回退。成功时确认位置前进到 max(acked_seq, ack_seq)，丢弃已确认
        前缀，并按本次实际删除的条数归还进程内共享容量——纯读取（确认
        位置不前进、无删除）与旧确认不释放任何容量，因此读不到新事件时
        不能靠补取腾出容量，生产方需等待真实确认后重试。返回确认位置与
        剩余未确认事件（按 seq 递增；内容与序号自入队起不变，重复补取
        结果相同），并把“已返回的最大 seq”推进到本次返回的末尾。
        未开启通道抛 ChannelMissing（404）；会话已结束抛
        SessionFinished（404）。与既有会话操作在会话锁内完整串行。
        """
        with session.lock:
            if session.closed:
                raise SessionFinished(session.session_id)
            channel = session.channel
            if channel is None:
                raise ChannelMissing(session.session_id)
            if ack_seq > channel.max_returned_seq:
                raise AckOutOfRange(ack_seq, channel.max_returned_seq)
            new_acked = channel.acked_seq
            if ack_seq > new_acked:
                new_acked = ack_seq
            # 丢弃新确认的前缀；仅本次实际删除的条数释放共享容量，
            # 纯读取与旧确认（无删除）不释放。剩余事件保持入队时的
            # 内容与序号、按 seq 递增。
            deleted = 0
            if channel.events and channel.events[0]["seq"] <= new_acked:
                kept = [e for e in channel.events if e["seq"] > new_acked]
                deleted = len(channel.events) - len(kept)
                channel.events = kept
            channel.acked_seq = new_acked
            if deleted:
                self._channel_capacity.release(deleted)
            remaining = [dict(e) for e in channel.events]
            if remaining:
                channel.max_returned_seq = remaining[-1]["seq"]
            return {
                "session_id": session.session_id,
                "scene_id": session.scene_id,
                "acked_seq": channel.acked_seq,
                "events": remaining,
            }

    def accept(
        self,
        session: PlaybackSession,
        load_record: Callable[[], Optional[SceneRecord]],
    ) -> Dict[str, Any]:
        """接纳场景当前快照：读取场景、校验、重算分轨、替换会话为完整操作。

        load_record 在会话锁内调用，因此读到的是“轮到本次接纳生效时”的
        场景快照；与同一会话的推进/接纳/结束串行。这样并发下迟到的请求
        不会把进入锁之前读到的旧快照回写到会话上。会话已结束抛
        SessionFinished（404）；场景不存在抛 SceneUnavailable；冲突或
        无解抛 AcceptConflict，会话保持不变。
        """
        with session.lock:
            if session.closed:
                raise SessionFinished(session.session_id)
            record = load_record()
            if record is None:
                raise SceneUnavailable(session.scene_id)
            if record.track_count != session.track_count:
                raise AcceptConflict(
                    f"轨道数不得变化：会话为 {session.track_count}，"
                    f"场景当前为 {record.track_count}"
                )

            # 游标前已开始（start_ms <= 游标）的字幕必须原样保留
            started = {
                s["id"]: s
                for s in session.subtitles
                if s["start_ms"] <= session.cursor_ms
            }
            new_by_id = {s["id"]: s for s in record.subtitles}
            for sid, old in started.items():
                new = new_by_id.get(sid)
                if new is None:
                    raise AcceptConflict(f"字幕 {sid!r} 已开始，不得删除")
                if _identity(new) != _identity(old):
                    raise AcceptConflict(
                        f"字幕 {sid!r} 已开始，其 id/speaker/起止时间/"
                        "locked_track 不得修改"
                    )
            # 其余字幕可改删或加入，但都必须在游标之后开始
            for new in record.subtitles:
                if new["start_ms"] <= session.cursor_ms and new["id"] not in started:
                    raise AcceptConflict(
                        f"字幕 {new['id']!r} 的开始时间不晚于游标 "
                        f"{session.cursor_ms}：新增/修改的字幕必须在游标之后开始"
                    )

            # 固定已开始字幕的原分轨，按新快照顺序与最少换轨、并列规则重算
            session_track = {
                s["id"]: t for s, t in zip(session.subtitles, session.tracks)
            }
            subs = [
                Subtitle(
                    id=s["id"],
                    speaker=s["speaker"],
                    start_ms=s["start_ms"],
                    end_ms=s["end_ms"],
                    locked_track=(
                        session_track[s["id"]]
                        if s["id"] in started
                        else s.get("locked_track")
                    ),
                )
                for s in record.subtitles
            ]
            tracks, changes = solve(subs, session.track_count)
            if tracks is None:
                raise AcceptConflict("固定已开始字幕的原分轨后，新快照无解")

            # 成功：整体替换会话快照与源版本（场景本身不变）。
            # hidden_speakers 是仅属于本会话的播放期设置，不在快照内，
            # 接纳后原样保留；active() 与事件推进自然按新字幕的 speaker 过滤。
            session.subtitles = [dict(s) for s in record.subtitles]
            session.tracks = list(tracks)
            session.changes = changes
            session.source_revision = record.revision
            return session.state()

    def finish(self, session_id: str) -> Dict[str, Any]:
        """结束会话：一次完整操作取回最终快照并移除注册表条目。

        完整操作的定序为：先在注册表中查到会话，再进入会话锁，在锁内
        置位 closed 墓碑并取出注册表条目，最后对轮到结束生效时的会话
        状态取快照。与同一会话的推进、事件推进、隐藏切换、接纳共用会话
        锁，因此与它们完整串行：

        - 快照只取自会话内存状态（session.state()），不读取场景存储，
          不会纳入尚未被接纳的场景修订；
        - 已取得会话引用但在锁上等待的在途请求，轮到它时看到 closed，
          一律抛 SessionFinished（404），不会在结束之后再生效一次；
        - 并发结束中第一个通过 closed 检查的请求成功，其余抛
          SessionFinished；注册表已无条目时抛 SessionMissing；
        - 不改动源场景、其他会话或此前返回的快照；不保存结束记录。

        返回的字典沿用创建会话（state()）的全部响应字段。
        """
        with self._registry_lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise SessionMissing(session_id)
        with session.lock:
            if session.closed:
                raise SessionFinished(session_id)
            session.closed = True
            # 结束会话清除补取通道（通道本就只是会话内存态）：
            # 归还该通道现存未确认事件占用的共享容量。closed 检查保证
            # 并发结束只有一个请求到达此处，因此剩余容量只释放一次。
            channel = session.channel
            if channel is not None:
                self._channel_capacity.release(len(channel.events))
                session.channel = None
            with self._registry_lock:
                # 注册表条目即本对象；以 closed 为准，防御性地按 id 移除
                self._sessions.pop(session_id, None)
            # state() 复制字幕与分轨：结束响应不受其后任何变化影响
            return session.state()
