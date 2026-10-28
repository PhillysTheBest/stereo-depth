# colab-bridge

Runs commands and notebooks on a free Colab T4 from a machine with no GPU, such as a
Claude cloud session. It does the same job as
[claude-colab](https://github.com/LakshmiSravyaVedantham/claude-colab), but it's
written from scratch so the code that gets a shell on our Colab VM is code we own.

```
Colab VM                                         Agent / laptop
┌───────────────────────────┐                    ┌────────────────────┐
│ colab_bridge.server       │                    │ colab-bridge CLI   │
│  127.0.0.1 only           │◄── Cloudflare ────►│                    │
│  /start /poll /kill       │    quick tunnel    │                    │
│  /put /get /health        │  (Fernet bodies)   │                    │
└───────────────────────────┘                    └────────────────────┘
```

## Use

1. Open `notebooks/colab_gpu_server.ipynb` in Colab, pick a T4 runtime, and click Run all.
   Set `REF` to a branch if the bridge isn't on `main` yet.
2. Copy the `colab-bridge://KEY@host` line it prints.
3. Where the agent runs:

```bash
uv tool install ./tools/colab_bridge        # or: uv run --project tools/colab_bridge colab-bridge ...
colab-bridge connect                        # paste the string on stdin
colab-bridge health
colab-bridge run "nvidia-smi"
colab-bridge notebook notebooks/cuda_sgm.ipynb --set REF=my-branch
```

`notebook` uploads the notebook, executes it on Colab with `nbconvert`, and downloads
`<name>.executed.ipynb` with every output. It exits non-zero if any cell raised.

## Security model

- **One secret.** The connection string holds a Fernet key generated fresh each session.
  Every request and response body is encrypted and authenticated with it, so Cloudflare
  sees only ciphertext and can't forge or alter a command. There's no bearer token that
  could leak in a header.
- **No replays.** Requests older than 2 minutes are rejected, and each accepted request
  is remembered so it can't be sent twice.
- **Quiet to strangers.** A request that fails authentication gets an empty 403.
- **Loopback only.** The server binds to 127.0.0.1, so the tunnel is the only way in.
- **Pinned tunnel binary.** cloudflared is a fixed release whose SHA-256 is checked
  before it runs, not "latest".
- **Secret stays out of argv.** `connect` reads the string from stdin or
  `COLAB_BRIDGE_URI` and saves it with mode 0600.

What it doesn't protect against: anyone holding the connection string has a shell on
that Colab VM, and Google can see everything on its own hardware. Disconnect and delete
the runtime to revoke access.

## Limits

Free Colab sessions stop after about 90 minutes idle or 12 hours total. Files move in
one request each, 50 MB at most. Long commands are fine because they run as background
jobs that the client polls.
