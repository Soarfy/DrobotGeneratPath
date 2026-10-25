import os

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

import isaacsim
from isaacsim import SimulationApp

print("starting SimulationApp ...", flush=True)
# This machine has both an RTX 4070 and an Intel iGPU. Kit's default
# "multiGpu/enabled" pulls both adapters into the renderer and the Vulkan
# device init segfaults, so pin everything to the discrete GPU.
app = SimulationApp({
    "headless": True,
    "multi_gpu": False,
    "extra_args": [
        "--/renderer/multiGpu/enabled=False",
        "--/renderer/activeGpu=0",
        "--/physics/cudaDevice=0",
        # 32 hardware threads on this i9-14900K; Kit's default of one worker
        # per logical core hits an access violation during startup teardown.
        "--/plugins/carb.tasking.plugin/threadCount=8",
        "--/plugins/omni.tbb.globalcontrol/maxThreadCount=8",
        "--/rtx/materialDb/syncLoads=True",
        "--/rtx/hydra/materialSyncLoads=True",
        "--/omni.kit.plugin/syncUsdLoads=True",
        "--/app/fastShutdown=False",
        # Pure headless + no RTX: skip renderer/viewport startup entirely. The
        # crash happens in a renderer worker thread just after "app ready",
        # so a compute-only Kit avoids it.
        "--/renderer/enabled=False",
        "--/exts/omni.kit.viewport.utility/enabled=False",
        "--/exts/omni.kit.renderer.core/enabled=False",
    ],
})

# Write a marker file rather than trusting stdout: Kit installs its own
# logging handler that can swallow prints from the main thread.
from pathlib import Path as _P

_P("out/isaac_booted.marker").write_text("booted", encoding="utf-8")

import omni.usd

stage = omni.usd.get_context().get_stage()
_P("out/isaac_stage.marker").write_text(str(stage), encoding="utf-8")
print("STAGE:", stage, flush=True)
print("ISAAC_READY", flush=True)

# Skip app.close(): the teardown path is what crashes on this machine.
import os as _os
_os._exit(0)
