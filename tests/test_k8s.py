"""Kubernetes sandbox tests that need no cluster: the pod manifest, path translation, and the whole
gateway -> PodHost -> pod server chain with the K8s API server replaced by an in-process transport."""

import json
import sys
from pathlib import Path

import httpx
import pytest

from llm_gateway.agent import podserver
from llm_gateway.agent.hosting import HostError
from llm_gateway.agent.k8s import PodHost, PodManager, host_path, pod_name
from llm_gateway.config import K8sConfig


@pytest.mark.skipif(sys.platform != "win32", reason="Windows drive translation")
def test_windows_paths_map_to_docker_desktop_node():
    assert host_path(Path(r"D:\program\Github\x\data\agents\a1"), "/run/desktop/mnt/host") == \
        "/run/desktop/mnt/host/d/program/Github/x/data/agents/a1"


def test_pod_manifest_is_a_sandbox(tmp_path):
    m = PodManager(K8sConfig(agents_dir=tmp_path), {"GATEWAY_URL": "http://gw/v1"}).manifest("20261022-ABC_1")
    assert m["metadata"]["name"] == pod_name("20261022-ABC_1") == "agent-20261022-abc-1"
    spec = m["spec"]
    assert spec["automountServiceAccountToken"] is False          # no K8s credentials in the pod
    c = spec["containers"][0]
    assert c["volumeMounts"] == [{"name": "home", "mountPath": "/agent"}]
    assert spec["volumes"][0]["hostPath"]["path"].endswith("20261022-ABC_1")   # only its own folder
    assert c["resources"]["limits"] == {"cpu": "2", "memory": "2Gi"}
    assert {"name": "AGENT_ID", "value": "20261022-ABC_1"} in c["env"]


class PodRouter(httpx.AsyncBaseTransport):
    """Stands in for `kubectl proxy` + the API server: strips the pod proxy prefix and hands the
    request to that pod's app (one app per agent, like one pod per agent)."""

    def __init__(self, make_app):
        self.make_app = make_app
        self.pods: dict[str, httpx.ASGITransport] = {}

    async def handle_async_request(self, request):
        path = request.url.path            # /api/v1/namespaces/ns/pods/agent-x:8000/proxy/session
        pod = path.split("/pods/")[1].split(":")[0]
        if pod not in self.pods:
            self.pods[pod] = httpx.ASGITransport(app=self.make_app(pod))
        rest = path.split("/proxy", 1)[1] or "/"
        url = request.url.copy_with(path=rest)
        return await self.pods[pod].handle_async_request(httpx.Request(request.method, url, headers=request.headers,
                                                                       content=request.content))


async def test_pod_host_runs_each_agent_in_its_own_folder(tmp_path, monkeypatch):
    agents_dir = tmp_path / "agents"
    host = PodHost(K8sConfig(agents_dir=agents_dir), {}, "default")
    host.started, host.proxy.url = True, "http://kube-proxy"

    async def no_pod_needed(agent_id):
        return None
    monkeypatch.setattr(host.pods, "ensure", no_pod_needed)
    monkeypatch.setattr(host.pods, "status", lambda agent_id: _ready())

    def make_app(pod):   # what the pod would run, with /agent pointing at that agent's host folder
        agent_id = next(d.name for d in agents_dir.iterdir() if pod_name(d.name) == pod)
        monkeypatch.setattr(podserver, "ROOT", agents_dir / agent_id)
        return podserver.create_pod_app({"AGENT_ID": agent_id, "GATEWAY_URL": "http://gw/v1",
                                         "MODEL_CONTEXTS": json.dumps({"m": 32768})})
    host.client = httpx.AsyncClient(transport=PodRouter(make_app))

    a = await host.create("m", "default", None)
    b = await host.create("m", "plan", None)
    assert a["id"] != b["id"] and a["workspace"] == "/agent/workspace"
    assert Path(a["host_folder"]) == agents_dir / a["id"] / "workspace" and Path(a["host_folder"]).is_dir()
    assert {s["id"] for s in await host.list()} == {a["id"], b["id"]}

    updated = await host.update(a["id"], {"mode": "accept_edits", "title": "Read hermes-agent"})
    assert updated["mode"] == "accept_edits" and updated["title"] == "Read hermes-agent"
    # renaming a stopped agent edits its checkpoint without starting a pod
    monkeypatch.setattr(host.pods, "status", lambda agent_id: _stopped())
    assert (await host.update(b["id"], {"title": "Second"}))["title"] == "Second"
    monkeypatch.setattr(host.pods, "status", lambda agent_id: _ready())
    with pytest.raises(HostError) as e:
        await host.update(a["id"], {"workspace": "C:/"})        # a pod's folder is fixed
    assert e.value.status == 400
    with pytest.raises(HostError) as e:
        await host.permission(a["id"], "nope", podserver.Answer(True))
    assert e.value.status == 404                                # error from inside the pod, passed through

    await host.delete(b["id"])
    assert not (agents_dir / b["id"]).exists()
    with pytest.raises(HostError):
        await host.get(b["id"])


async def _ready():
    return "ready"


async def _stopped():
    return "none"
