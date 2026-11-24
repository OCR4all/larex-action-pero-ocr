# LAREX Action: pero-ocr

Layout analysis and text recognition for [LAREX](https://github.com/OCR4all/larex) with [pero-ocr](https://github.com/DCGM/pero-ocr) 0.7.0.
The Action detects text regions and lines, determines the reading order, recognizes the text, and returns PAGE XML.

## Setup

1. Copy `template.env` to `.env`, then set the dispatch secret and the origin of the LAREX server.

   ```bash
   cp template.env .env
   ```

2. Create the model directory and install engines. The following command downloads `pero_eu_cz_print_newspapers_2022-09-26`, the engine published by the pero-ocr authors for European printed newspapers. The archive has 363 MB and is verified against a pinned checksum.

   ```bash
   mkdir -p models
   ```

   ```bash
   MODEL_UID=$(id -u) MODEL_GID=$(id -g) docker compose run --rm --build models
   ```

   The processor only needs read access to the models afterwards.

3. Start the processor.

   ```bash
   docker compose up -d --build
   ```

   For NVIDIA GPUs, add the overlay. It needs an x86_64 host with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) and its CDI configuration.

   ```bash
   docker compose -f compose.yml -f compose.gpu.yml up -d --build
   ```

4. Adjust the URLs and `secretRef` in `action.yml` and add the definition in LAREX. The endpoint path has to start with one of `LAREX_ROUTE_PREFIXES`.

LAREX has to support incremental page results and run reports.

## Engines

Every directory in `MODEL_DIR` with a pero-ocr configuration is offered as an engine under its directory name:

```text
models/
├─ pero_eu_cz_print_newspapers_2022-09-26/
│  ├─ config.ini
│  ├─ ocr_engine.json
│  ├─ OCR_350000.pt        (GPU)
│  ├─ OCR_350000.pt.cpu    (CPU)
│  ├─ ParseNet_296000.pt
│  └─ ParseNet_296000.pt.cpu
└─ another_engine/
   └─ config.ini
```

- The configuration is `config.ini`, otherwise `config_cpu.ini`, otherwise the only `.ini` file of the directory.
- pero-ocr needs separate weights for the CPU, with the suffix `.cpu`. The processor selects the device, so `USE_CPU` in the configuration is ignored.
- Engines without layout detection can recognize existing lines, and engines without OCR can segment pages.
- Added or changed engines are picked up by the next run. A restart is not needed.
- The networks are loaded with PyTorch, which can execute code from model files. Only install engines from sources you trust.

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `engine` | first engine or `PERO_DEFAULT_ENGINE` | Engine installed on the processor. |
| `segmentation` | `auto` | What is segmented before the recognition, see below. |
| `recognition` | `true` | Recognize the text of all lines in scope. |
| `textIndex` | `1` | `TextEquiv` index of the recognized text. Index 0 usually holds the ground truth. |
| `minConfidence` | `0` | Leave out newly detected lines that were recognized with a lower confidence. Existing lines are always kept. |

| `segmentation` | Page | Selected regions | Selected lines |
| --- | --- | --- | --- |
| `auto` | Detect regions and lines if the page has no text regions, otherwise the lines of regions without lines | Detect the lines of regions without lines | Not segmented |
| `regions` | Replace all text regions and their lines | Not available | Not available |
| `lines` | Replace the lines of all text regions | Replace the lines | Not available |
| `keep` | Not segmented | Not segmented | Not segmented |

## Results

- Other regions, such as images and tables, stay in place and are moved behind the new text regions in the reading order.
- Pages and regions in which nothing is detected keep their annotations. Such pages are reported as skipped, such regions as a warning.
- Existing lines need no baseline. It is estimated from the polygon, so lines drawn as boxes in LAREX can be recognized. Other `TextEquiv` indices, words, and glyphs are not changed.
- The confidence of a line is the confidence of its weakest character.
- If the size of the image and the PAGE XML differ, coordinates are scaled. The page fails if the aspect ratios do not match.
- A page fails without affecting the others when its image is missing, unreadable, or larger than `PERO_MAX_IMAGE_MEGAPIXELS`.
- Coordinates refer to the stored pixels. EXIF orientation is ignored, as in LAREX.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `NAME`, `PORT` | `pero-ocr`, `9000` | Suffix of the container name and published port. Compose only. |
| `MODEL_DIR` | `./models` | Host directory with the engines. Compose only. |
| `LAREX_PROCESSOR_ID` | | Has to match `id` in `action.yml`. |
| `LAREX_DISPATCH_HMAC_SECRET` | | Secret behind `secretRef`. |
| `LAREX_ALLOWED_CALLBACK_ORIGINS` | | JSON list with the origins of the LAREX server. |
| `LAREX_ROUTE_PREFIXES` | `[]` | JSON list of path prefixes. |
| `LAREX_SDK_TRANSPORT_DEBUG` | `false` | Log the requests to LAREX. |
| `ACTION_DEBUG` | `false` | Verbose logs, including the output of pero-ocr. |
| `ACTION_MAX_CONCURRENT_RUNS` | `1` | Runs that are processed at the same time. Further runs wait. Every concurrent run uses its own worker process and its own copy of the models. |
| `ACTION_LOADING_TIMEOUT` | `180` | Seconds until loading the engine fails the run. |
| `ACTION_PROCESSING_TIMEOUT` | `600` | Seconds per page. |
| `ACTION_TIMEOUT_MODE` | `fail_page` | `fail_page` continues with the next page, `fail_action` stops the run. |
| `ACTION_IDLE_TIMEOUT` | `600` | Seconds for which a worker keeps its engine in memory after a run, `0` to release it immediately. |
| `PERO_MODEL_DIR` | `/models` | Directory with the engines inside the container. |
| `PERO_DEFAULT_ENGINE` | first engine | Preselected engine. |
| `PERO_DEVICE` | `auto` | `auto`, `cpu`, `cuda`, or `cuda:N`. With `auto`, the GPU image logs a warning when it has to use the CPU. With `cuda`, the run fails instead. |
| `PERO_CPU_THREADS` | up to 4 | Threads per worker. |
| `PERO_BATCH_PIXELS` | 3840 (CPU), 15360 (GPU) | Total width of the lines recognized in one batch. Larger batches need more GPU memory. |
| `PERO_MAX_IMAGE_MEGAPIXELS` | `200` | Largest accepted image. |
| `PERO_PREFETCH_PAGES` | `2` | Pages that are downloaded ahead. |

## How it works

- The web process holds no models. Runs use a worker process that loads the engine and stays warm for following runs with the same engine.
- Only the required networks are loaded, for example no layout network for `keep`.
- The next pages are downloaded and the previous result is uploaded while a page is processed.
- Lines are batched by width. When GPU memory runs out, the batch is halved and repeated.
- Cancellation and timeouts stop the worker process. After a crash the page fails and a new worker continues with the next page. Three crashes in a row stop the run.
- When the processor is stopped, active runs are reported as failed.

### Differences from pero-ocr

The detection and recognition networks, their pre- and postprocessing, and the greedy decoding are those of pero-ocr. These parts are replaced or adapted:

- The decoding runs on the device of the network, so that the full network output is not copied for every line. Engines with `RUN_DECODER` or a transformer network use the implementation of pero-ocr.
- The mapping of line crops, written in slow object mode Numba, is vectorized with identical results.
- The process pool that pero-ocr opens for every engine is never used and is not started.
- Regions whose outline falls into several parts stop pero-ocr with Shapely 2. They become separate regions.
- Lines wider than 3840 pixels are cut off in pero-ocr. Here they are recognized up to 16384 pixels.
- Lines of given regions are cut at the region outline. Small remainders of lines that mainly belong to a neighbour are dropped, and overlapping regions do not receive the same line twice.
- The scale that pero-ocr carries over from page to page is reset for every run, and once more if a page seems empty.
- If sorting the regions fails, they stay in the order of detection and the page gets a warning.
- The random numbers of the region detection are seeded, so that repeated runs give the same result.
- `STRETCH_LINES = max` stops pero-ocr 0.7.0 with an error and is rejected when the engine is loaded.

### Pinned versions

- pero-ocr 0.7.0 is the latest release. Its development branch is not released and depends on NumPy 1 and AGPL licensed packages.
- Dependencies that are only used for training or for other entry points are excluded in `pyproject.toml`, and `opencv-python` is replaced with its headless variant.
- SciPy is limited to versions before 2, because pero-ocr uses `scipy.ndimage.morphology`.
- The GPU image uses PyTorch for CUDA 12.9.

## Development

```bash
uv sync --extra cpu --extra test
```

```bash
uv run --no-sync pytest
```

The tests replace the networks of pero-ocr with small stand-ins and need no downloaded engine.

## License

MIT. pero-ocr is licensed under the BSD 3-Clause License. Engines have their own terms, so check them before use.
