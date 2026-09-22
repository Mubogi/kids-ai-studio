# Example input frames

Drop one image per scene here to drive **image-to-video** mode. PNGs in this
directory are gitignored, so your own photos never get committed.

```bash
# one image per scene, in order
python -m kidvid --video-backend mock --music-backend mock \
    --image examples/scene1.png \
    --image examples/scene2.png \
    --image examples/scene3.png \
    --image examples/scene4.png \
    "my dog goes to space"
```

Scenes without a matching image fall back to text-to-video, so a partial list
works fine. Images should roughly match the output aspect ratio (832x480 by
default) — anything else is scaled and letterboxed to fit.
