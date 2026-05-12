# Marathon Route Extraction Web App

FastAPI backend + single-page HTML frontend for marathon route mask extraction.

## Project Layout

```
capstone-project/
├── app.py                              # FastAPI entrypoint
├── requirements.txt                    # Python dependencies
├── scripts/
│   └── visualize_component_filter.py   # Optional debug visualization utility
├── src/
│   ├── __init__.py
│   ├── gpx_conversion/
│   │   └── gpx_converter.py            # Pixel path -> GPX conversion
│   └── marathon_route_extraction/
│       ├── anchor_filter.py            # OCR anchor filtering
│       ├── component_filter.py         # Connected-component filtering
│       ├── config.py                   # Runtime config
│       ├── model.py                    # U-Net model/inference
│       └── path_extractor.py           # Skeleton + ordered path extraction
├── static/
│   └── index.html                      # 5-stage demo UI
└── weights/
    └── model_best.pt                   # Trained model weights
```

## Run

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Ensure model weights exist at `weights/model_best.pt`

3. Start server

```bash
python app.py
```

4. Open browser

```text
http://localhost:8010
```

## API Endpoints

- `POST /api/predict`: image upload and mask prediction
- `POST /api/postprocess`: component filtering + skeletonization
- `POST /api/extract_path`: ordered path extraction from start to end
- `POST /api/convert_gpx`: GPX conversion (placeholder)
- `GET /api/health`: health check

## Utility Script

Visualize component filtering result:

```bash
python scripts/visualize_component_filter.py --input-mask sample_route.png --min-component-area 130
```

## Notes

- If startup fails with address-in-use on port 8010, change `PORT` in `src/marathon_route_extraction/config.py`.
- GPX conversion now lives in `src/gpx_conversion/gpx_converter.py`.
