"""补取通道共享容量：CHANNEL_EVENT_LIMIT 控制单进程内全部会话的
未确认事件总量。

覆盖：
- 配置：默认 1024；正整数可覆盖（Compose 经环境变量注入）；未设置取
  默认；空串/0、负数、小数、布尔样式、空白与其他非数字一律 ValueError
  （启动拒绝，lifespan 下 TestClient 进入即失败；uvicorn 子进程非零退出）；
- 申请：普通推进、事件推进、隐藏切换按本次实际生成的事件数占用共享
  容量（跨会话合计）；不足立即 503，且不推进游标、不改变隐藏集合、
  不消费 seq、不部分入队、不占用容量；
- 无通道不占容量；零事件操作（等时推进、倒退 422、无变化切换）在满额
  时仍可执行；一批事件整体申请（要么全部入队要么 503）；
- 释放：仅确认并实际删除前缀时按删除条数释放；纯读取与旧确认不释放；
  结束释放剩余队列；重复/并发结束不重复释放；释放后原请求可重试；
- 并发：不同会话并发生产不能突破总额；生产、确认、结束混合同步运行
  不死锁、不泄漏容量（全部结束后 in_use 归零）。

测试场景 scene_payload：s_i=[i*1000+100, i*1000+800)（如
s1=[1100,1800)、s2=[2100,2800)），说话人互不相同。边界时刻不与整千
采样重合，故每次 (i-1)*1000 -> i*1000 的推进恰好产生 2 条事件，
游标 1500 处仅 s1 活动。
"""
import os
import subprocess
import sys
import threading

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from app import main as main_module  # noqa: E402
from app.main import app, channel_event_limit  # noqa: E402
from app.playback import (  # noqa: E402
    DEFAULT_CHANNEL_EVENT_LIMIT,
    channel_event_limit_from_env,
)


def scene_payload(n=6):
    """6 条互不重叠、彼此留空的单轨字幕；说话人互不相同。"""
    return {
        "track_count": 2,
        "expected_revision": 0,
        "subtitles": [
            {
                "id": f"s{i}",
                "speaker": f"Sp{i}",
                "start_ms": i * 1000 + 100,
                "end_ms": i * 1000 + 800,
            }
            for i in range(1, n + 1)
        ],
    }


@pytest.fixture
def limit_client(_isolated_scene_db, monkeypatch):
    """以小容量 CHANNEL_EVENT_LIMIT 构造隔离 client 的工厂。

    显式依赖 conftest 的 _isolated_scene_db（其末尾重置单例），随后再
    设置容量环境变量并重置一次，确保管理器按本用例的限额惰性创建。
    """
    def _factory(limit):
        monkeypatch.setenv("CHANNEL_EVENT_LIMIT", str(limit))
        main_module.reset_playback()
        assert channel_event_limit() == limit
        return TestClient(app)

    return _factory


def put_scene(client, scene_id="scene-limit"):
    resp = client.put(f"/scenes/{scene_id}", json=scene_payload())
    assert resp.status_code == 200


def make_session(client, scene_id="scene-limit", open=True):
    sid = client.post(
        "/sessions", json={"scene_id": scene_id}
    ).json()["session_id"]
    if open:
        assert client.post(f"/sessions/{sid}/channel").status_code == 200
    return sid


def make_scene_session(client, scene_id="scene-limit"):
    put_scene(client, scene_id)
    return make_session(client, scene_id)


def fetch(client, sid, ack_seq):
    return client.post(
        f"/sessions/{sid}/channel/fetch", json={"ack_seq": ack_seq}
    )


# ---------------------------------------------------------------------------
# 配置解析：默认 1024、正整数合法、非法拒绝启动
# ---------------------------------------------------------------------------


def test_default_limit_is_1024():
    assert DEFAULT_CHANNEL_EVENT_LIMIT == 1024
    # 未设置（None）取默认；显式空串/空白属非法配置（见拒绝用例）
    assert channel_event_limit_from_env(None) == 1024


@pytest.mark.parametrize("raw", ["1", "2", "1024", "999999999999999999999"])
def test_limit_parses_positive_integer(raw):
    assert channel_event_limit_from_env(raw) == int(raw)


@pytest.mark.parametrize(
    "raw",
    [
        "0", "-1", "-0", "1.5", "1e3", "0x10", "true", "false", "yes",
        "1024 ", " 1024", "1_024", "１２", "abc", "", "   ",
    ],
)
def test_limit_rejects_non_positive_integer(raw):
    with pytest.raises(ValueError):
        channel_event_limit_from_env(raw)


def test_get_playback_uses_env_limit(limit_client):
    client = limit_client(3)
    sid = make_scene_session(client)
    mgr = main_module.get_playback()
    assert mgr.channel_event_limit == 3
    # 0->2000 跨 s1=[1100,1800)：enter@1100、exit@1800 两条
    resp = client.post(f"/sessions/{sid}/events", json={"time_ms": 2000})
    assert resp.status_code == 200
    assert mgr.channel_events_in_use == 2


def test_invalid_limit_refuses_startup(monkeypatch, tmp_path):
    """lifespan 启动即校验：非法配置下服务不接受任何请求。"""
    monkeypatch.setenv("CHANNEL_EVENT_LIMIT", "0")
    monkeypatch.setenv("SCENES_DB_PATH", str(tmp_path / "x.db"))
    main_module.reset_playback()
    with pytest.raises(Exception):
        with TestClient(app) as client:
            client.get("/health")
    # 未进入 lifespan 而直接取管理器同样抛 ValueError
    main_module.reset_playback()
    with pytest.raises(ValueError):
        main_module.get_playback()


def test_invalid_limit_rejects_uvicorn_boot(tmp_path):
    """端到端：非法 CHANNEL_EVENT_LIMIT 下 uvicorn 进程以非零码退出。"""
    env = dict(os.environ)
    env["CHANNEL_EVENT_LIMIT"] = "abc"
    env["SCENES_DB_PATH"] = str(tmp_path / "boot.db")
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import uvicorn;"
                "uvicorn.run('app.main:app', host='127.0.0.1',"
                " port=8123, log_level='error')"
            ),
        ],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode != 0
    assert "CHANNEL_EVENT_LIMIT" in (proc.stderr + proc.stdout)


# ---------------------------------------------------------------------------
# 满额 503：原子性——游标/隐藏集合/seq/部分入队/容量均不变
# ---------------------------------------------------------------------------


def test_events_advance_over_limit_returns_503_and_changes_nothing(limit_client):
    client = limit_client(3)
    sid = make_scene_session(client)

    # 先占 2 条：0->2000 跨 s1=[1100,1800) 的 enter@1100/exit@1800
    assert client.post(
        f"/sessions/{sid}/events", json={"time_ms": 2000}
    ).status_code == 200

    # 游标 2000 -> 3000：跨 s2=[2100,2800) 两条，仅剩 1：503
    resp = client.post(f"/sessions/{sid}/events", json={"time_ms": 3000})
    assert resp.status_code == 503
    detail = resp.json()["detail"]
    assert detail["requested"] == 2
    assert detail["channel_event_limit"] == 3
    assert detail["in_use"] == 2

    # 游标未推进：仍停在 2000（等时推进零事件，满额仍 200）
    state = client.post(f"/sessions/{sid}/advance", json={"time_ms": 2000})
    assert state.json()["cursor_ms"] == 2000
    # seq 未消费：补取仍只有 seq 1、2 两条
    body = fetch(client, sid, 0).json()
    assert [e["seq"] for e in body["events"]] == [1, 2]
    # 容量未被部分占用
    assert main_module.get_playback().channel_events_in_use == 2


def test_503_request_retries_after_ack_release(limit_client):
    client = limit_client(3)
    sid = make_scene_session(client)
    client.post(f"/sessions/{sid}/events", json={"time_ms": 2000})  # 2 条
    assert client.post(
        f"/sessions/{sid}/events", json={"time_ms": 3000}
    ).status_code == 503

    # 确认已返回的 2 条（先补取过才能确认）：释放 2，容量清空
    assert fetch(client, sid, 0).status_code == 200
    assert fetch(client, sid, 2).status_code == 200
    assert main_module.get_playback().channel_events_in_use == 0

    # 原请求重试成功：s2 两条入队。seq 是通道内单调连续计数器，确认
    # 删除前缀后也不重置，故从 3、4 继续（事件内容与本次操作一致）。
    resp = client.post(f"/sessions/{sid}/events", json={"time_ms": 3000})
    assert resp.status_code == 200
    body = fetch(client, sid, 2).json()
    assert [(e["seq"], e["time_ms"], e["type"]) for e in body["events"]] == [
        (3, 2100, "enter"),
        (4, 2800, "exit"),
    ]


def test_plain_advance_503_does_not_move_cursor(limit_client):
    client = limit_client(2)
    sid = make_scene_session(client)
    # 普通推进到 2000 跨 s1 的两条恰好满额
    assert client.post(
        f"/sessions/{sid}/advance", json={"time_ms": 2000}
    ).status_code == 200
    # 下一次普通推进到 3000 需要 2（s2 enter/exit），容量为 0：503，游标不动
    resp = client.post(f"/sessions/{sid}/advance", json={"time_ms": 3000})
    assert resp.status_code == 503
    state = client.post(f"/sessions/{sid}/advance", json={"time_ms": 2000})
    assert state.json()["cursor_ms"] == 2000


def test_hidden_toggle_503_keeps_hidden_set_and_seq(limit_client):
    client = limit_client(1)
    # 未开启通道时推进到 1500（推进不占共享容量）：游标处仅 s1 活动
    put_scene(client)
    sid = make_session(client, open=False)
    client.post(f"/sessions/{sid}/advance", json={"time_ms": 1500})
    client.post(f"/sessions/{sid}/channel")

    # 隐藏 s1 的说话人 Sp1：产生 1 条 exit，恰好占满唯一容量
    assert client.post(
        f"/sessions/{sid}/hidden_speakers",
        json={"hidden_speakers": ["Sp1"]},
    ).status_code == 200

    # 恢复显示需 1 条 enter，容量为 0：503
    resp = client.post(
        f"/sessions/{sid}/hidden_speakers", json={"hidden_speakers": []}
    )
    assert resp.status_code == 503
    # 隐藏集合未变：active 仍排除 Sp1（游标 1500 处无可见字幕）
    body = client.post(
        f"/sessions/{sid}/advance", json={"time_ms": 1500}
    ).json()
    assert body["active"] == []
    # seq 未消费：通道中只有 seq 1
    assert [e["seq"] for e in fetch(client, sid, 0).json()["events"]] == [1]


def test_batch_is_all_or_nothing(limit_client):
    """一批多条事件整体申请：容量不够整批时一条都不入队。"""
    client = limit_client(3)
    sid = make_scene_session(client)
    # 占 2：窗口 (0,2000] 含 s1 enter@1100/exit@1800
    client.post(f"/sessions/{sid}/events", json={"time_ms": 2000})
    # 一次推进 2000->4000 跨 s2、s3 共 4 条（2100/2800、3100/3800），
    # 剩 1：整批 503，无部分入队
    resp = client.post(f"/sessions/{sid}/events", json={"time_ms": 4000})
    assert resp.status_code == 503
    assert resp.json()["detail"]["requested"] == 4
    body = fetch(client, sid, 0).json()
    assert [e["seq"] for e in body["events"]] == [1, 2]
    assert main_module.get_playback().channel_events_in_use == 2


# ---------------------------------------------------------------------------
# 无通道不占容量；零事件操作在满额时仍可执行
# ---------------------------------------------------------------------------


def test_sessions_without_channel_consume_no_capacity(limit_client):
    client = limit_client(1)
    put_scene(client)
    # 两个未开启通道的会话：产生多少事件都不占容量
    for _ in range(2):
        sid = make_session(client, open=False)
        assert client.post(
            f"/sessions/{sid}/events", json={"time_ms": 6000}
        ).status_code == 200
    assert main_module.get_playback().channel_events_in_use == 0

    # 开启通道的会话立即受满额约束：第一批需要 2 而容量仅 1
    sid = make_session(client, open=True)
    resp = client.post(f"/sessions/{sid}/events", json={"time_ms": 2000})
    assert resp.status_code == 503


def test_zero_event_operations_still_work_when_full(limit_client):
    client = limit_client(2)
    sid = make_scene_session(client)
    client.post(f"/sessions/{sid}/events", json={"time_ms": 2000})
    assert main_module.get_playback().channel_events_in_use == 2  # 满额

    # 等时事件推进：零事件，200，不占容量
    resp = client.post(f"/sessions/{sid}/events", json={"time_ms": 2000})
    assert resp.status_code == 200
    assert resp.json()["events"] == []
    # 等时普通推进同样可执行
    assert client.post(
        f"/sessions/{sid}/advance", json={"time_ms": 2000}
    ).status_code == 200
    # 隐藏切换到不含任何活动字幕说话人的集合：零事件，200
    resp = client.post(
        f"/sessions/{sid}/hidden_speakers",
        json={"hidden_speakers": ["Nobody"]},
    )
    assert resp.status_code == 200
    assert resp.json()["events"] == []
    # 再次提交同一集合：无变化，仍零事件
    resp = client.post(
        f"/sessions/{sid}/hidden_speakers",
        json={"hidden_speakers": ["Nobody"]},
    )
    assert resp.status_code == 200
    assert resp.json()["events"] == []

    # 倒退仍是 422（而非 503），且不占容量
    resp = client.post(f"/sessions/{sid}/events", json={"time_ms": 1000})
    assert resp.status_code == 422
    assert main_module.get_playback().channel_events_in_use == 2


def test_accept_does_not_consume_or_release_capacity(limit_client):
    client = limit_client(2)
    sid = make_scene_session(client)
    client.post(f"/sessions/{sid}/events", json={"time_ms": 2000})
    assert main_module.get_playback().channel_events_in_use == 2
    # 接纳（相同快照，无修订冲突）不触碰容量
    assert client.post(f"/sessions/{sid}/accept").status_code == 200
    assert main_module.get_playback().channel_events_in_use == 2
    # 接纳后生产依旧 503
    assert client.post(
        f"/sessions/{sid}/events", json={"time_ms": 3000}
    ).status_code == 503


# ---------------------------------------------------------------------------
# 容量在会话间共享
# ---------------------------------------------------------------------------


def test_capacity_shared_across_sessions(limit_client):
    client = limit_client(3)
    s1 = make_scene_session(client, "scene-shared-a")
    # s1 占 2
    client.post(f"/sessions/{s1}/events", json={"time_ms": 2000})

    s2 = make_scene_session(client, "scene-shared-b")
    # s2 尝试占 2，只剩 1：503
    resp = client.post(f"/sessions/{s2}/events", json={"time_ms": 2000})
    assert resp.status_code == 503
    assert resp.json()["detail"]["in_use"] == 2

    # s2 的等时推进不受影响
    assert client.post(
        f"/sessions/{s2}/events", json={"time_ms": 0}
    ).status_code == 200

    # s1 确认释放 2 后，s2 原请求重试成功
    assert fetch(client, s1, 0).status_code == 200
    assert fetch(client, s1, 2).status_code == 200
    assert client.post(
        f"/sessions/{s2}/events", json={"time_ms": 2000}
    ).status_code == 200
    assert main_module.get_playback().channel_events_in_use == 2


# ---------------------------------------------------------------------------
# 释放语义：读取/旧确认不释放；结束释放剩余且仅一次
# ---------------------------------------------------------------------------


def test_pure_fetch_does_not_release_capacity(limit_client):
    client = limit_client(2)
    sid = make_scene_session(client)
    client.post(f"/sessions/{sid}/events", json={"time_ms": 2000})
    assert main_module.get_playback().channel_events_in_use == 2

    # 多次纯读取（ack 0，未确认任何前缀）：容量不变，仍然满额
    for _ in range(3):
        body = fetch(client, sid, 0)
        assert body.status_code == 200
        assert len(body.json()["events"]) == 2
    assert main_module.get_playback().channel_events_in_use == 2
    assert client.post(
        f"/sessions/{sid}/events", json={"time_ms": 3000}
    ).status_code == 503


def test_old_ack_does_not_release_capacity(limit_client):
    client = limit_client(3)
    sid = make_scene_session(client)
    client.post(f"/sessions/{sid}/events", json={"time_ms": 2000})  # 2 条
    assert fetch(client, sid, 0).status_code == 200
    assert fetch(client, sid, 2).status_code == 200  # 确认 2，释放 2
    client.post(f"/sessions/{sid}/events", json={"time_ms": 3000})  # 再 2
    assert fetch(client, sid, 2).status_code == 200  # 先看到 seq 3、4
    assert main_module.get_playback().channel_events_in_use == 2

    # 旧确认（ack 2 < 当前已确认位置 4）不回退、不释放
    body = fetch(client, sid, 2).json()
    assert body["acked_seq"] == 2
    assert [e["seq"] for e in body["events"]] == [3, 4]
    assert main_module.get_playback().channel_events_in_use == 2

    # 部分确认：只释放实际删除的 1 条
    assert fetch(client, sid, 3).json()["acked_seq"] == 3
    assert main_module.get_playback().channel_events_in_use == 1


def test_finish_releases_remaining_queue_once(limit_client):
    client = limit_client(3)
    sid = make_scene_session(client)
    client.post(f"/sessions/{sid}/events", json={"time_ms": 2000})  # 占 2
    mgr = main_module.get_playback()
    assert mgr.channel_events_in_use == 2

    assert client.post(f"/sessions/{sid}/finish").status_code == 200
    assert mgr.channel_events_in_use == 0
    # 重复结束（会话已不存在）：404，不触碰容量
    assert client.post(f"/sessions/{sid}/finish").status_code == 404
    assert mgr.channel_events_in_use == 0

    # 释放出的容量可被新会话使用
    sid2 = make_scene_session(client, "scene-after-finish")
    assert client.post(
        f"/sessions/{sid2}/events", json={"time_ms": 2000}
    ).status_code == 200


def test_finish_with_empty_or_no_channel_releases_nothing(limit_client):
    client = limit_client(2)
    # 无通道会话结束
    put_scene(client)
    sid = make_session(client, open=False)
    client.post(f"/sessions/{sid}/events", json={"time_ms": 2000})
    assert client.post(f"/sessions/{sid}/finish").status_code == 200
    assert main_module.get_playback().channel_events_in_use == 0

    # 有通道但队列已全部确认后结束
    sid2 = make_scene_session(client, "scene-empty-channel")
    client.post(f"/sessions/{sid2}/events", json={"time_ms": 2000})
    fetch(client, sid2, 0)
    fetch(client, sid2, 2)
    assert main_module.get_playback().channel_events_in_use == 0
    assert client.post(f"/sessions/{sid2}/finish").status_code == 200
    assert main_module.get_playback().channel_events_in_use == 0


def test_finish_unblocks_other_session_producer(limit_client):
    client = limit_client(2)
    s1 = make_scene_session(client, "scene-blocker")
    client.post(f"/sessions/{s1}/events", json={"time_ms": 2000})  # 占满
    s2 = make_scene_session(client, "scene-blocked")
    assert client.post(
        f"/sessions/{s2}/events", json={"time_ms": 2000}
    ).status_code == 503

    # 结束 s1 释放全部剩余队列后，s2 无需确认即可重试成功
    assert client.post(f"/sessions/{s1}/finish").status_code == 200
    assert client.post(
        f"/sessions/{s2}/events", json={"time_ms": 2000}
    ).status_code == 200
    assert main_module.get_playback().channel_events_in_use == 2


# ---------------------------------------------------------------------------
# 并发：总额不可突破；生产/确认/结束混合同步，不死锁、不泄漏
# ---------------------------------------------------------------------------


def test_concurrent_producers_never_exceed_limit(limit_client):
    limit = 6
    client = limit_client(limit)
    # 8 个会话各尝试一次跨 2 条事件的事件推进；每个会话独立场景
    sessions = [
        make_scene_session(client, f"scene-conc-{i}") for i in range(8)
    ]

    statuses = []
    succeeded = []
    lock = threading.Lock()
    barrier = threading.Barrier(8)

    def worker(local_sid):
        local = TestClient(app)
        barrier.wait()
        resp = local.post(
            f"/sessions/{local_sid}/events", json={"time_ms": 2000}
        )
        with lock:
            statuses.append(resp.status_code)
            if resp.status_code == 200:
                succeeded.append(local_sid)

    threads = [
        threading.Thread(target=worker, args=(sid,)) for sid in sessions
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # 每次申请 2、上限 6：恰好 3 个成功，其余 503
    assert sorted(statuses) == [200] * 3 + [503] * 5
    # 在用容量绝不超过上限，且等于成功次数 * 2
    mgr = main_module.get_playback()
    assert mgr.channel_events_in_use == limit
    assert len(succeeded) == 3
    blocked = [sid for sid in sessions if sid not in succeeded]
    # 成功方补取内容完整；落败方通道为空（游标仍为 0，未消费 seq）
    for sid in succeeded:
        assert [e["seq"] for e in fetch(client, sid, 0).json()["events"]] == [
            1, 2
        ]
    for sid in blocked:
        assert fetch(client, sid, 0).json()["events"] == []
    # 释放一个成功会话后，落败方原请求可重试成功
    fetch(client, succeeded[0], 2)
    retried = client.post(
        f"/sessions/{blocked[0]}/events", json={"time_ms": 2000}
    )
    assert retried.status_code == 200
    assert main_module.get_playback().channel_events_in_use == limit


def test_mixed_produce_ack_finish_is_deadlock_and_leak_free(limit_client):
    limit = 4
    client = limit_client(limit)
    slot_lock = threading.Lock()
    # 每个槽位固定一个场景，持有“当前会话 id + 已推进到的时刻”；
    # finisher 结束会话后用同场景重建，保持持续的生产/确认流量。
    slots = []
    for i in range(6):
        sid = make_scene_session(client, f"scene-mix-{i}")
        slots.append({"scene": f"scene-mix-{i}", "sid": sid, "time": 0})

    stop = threading.Event()
    barrier = threading.Barrier(3)

    def producer():
        barrier.wait()
        local = TestClient(app)
        idx = 0
        while not stop.is_set():
            with slot_lock:
                slot = slots[idx % len(slots)]
                idx += 1
                sid, old_time = slot["sid"], slot["time"]
            new_time = min(old_time + 1000, 6000)
            if new_time == old_time:
                continue
            resp = local.post(
                f"/sessions/{sid}/events", json={"time_ms": new_time}
            )
            # 仅成功才推进槽位时刻；503（容量不足）/404（已被结束）/
            # 422（游标回退，槽位已重建）都留给后续循环重试。
            if resp.status_code == 200:
                with slot_lock:
                    if slot["sid"] == sid:
                        slot["time"] = new_time

    def acker():
        barrier.wait()
        local = TestClient(app)
        idx = 0
        while not stop.is_set():
            with slot_lock:
                sid = slots[idx % len(slots)]["sid"]
                idx += 1
            body = fetch(local, sid, 0)
            if body.status_code == 200 and body.json()["events"]:
                top = body.json()["events"][-1]["seq"]
                # 确认全部已返回事件（按删除条数释放容量）
                fetch(local, sid, top)

    def finisher():
        barrier.wait()
        local = TestClient(app)
        idx = 0
        while not stop.is_set():
            with slot_lock:
                slot = slots[idx % len(slots)]
                idx += 1
                old_sid, scene = slot["sid"], slot["scene"]
            if local.post(f"/sessions/{old_sid}/finish").status_code != 200:
                continue
            resp = local.post(
                "/sessions", json={"scene_id": scene}
            )
            if resp.status_code != 200:
                continue
            new_sid = resp.json()["session_id"]
            if local.post(f"/sessions/{new_sid}/channel").status_code != 200:
                continue
            with slot_lock:
                # 仅在槽位仍指向被我们结束的会话时替换（正常串行下成立）
                if slot["sid"] == old_sid:
                    slot["sid"] = new_sid
                    slot["time"] = 0

    threads = [
        threading.Thread(target=producer),
        threading.Thread(target=acker),
        threading.Thread(target=finisher),
    ]
    for t in threads:
        t.start()
    # 2 秒混合压力足以暴露锁序死锁
    stop.wait(2.0)
    stop.set()
    for t in threads:
        t.join(timeout=10)
        assert not t.is_alive(), "生产/确认/结束并发出现死锁"

    # 结束全部当前会话后，共享容量必须归零（无泄漏、无重复释放）
    with slot_lock:
        live = [slot["sid"] for slot in slots]
    for sid in live:
        client.post(f"/sessions/{sid}/finish")
    assert main_module.get_playback().channel_events_in_use == 0
