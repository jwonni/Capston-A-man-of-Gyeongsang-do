# 흐름 요약
1. OCR로 마라톤 이미지에서 텍스트 검출 → 주요 랜드마크 후보 추출
2. OCR 텍스트 → 위/경도 지오코딩 (카카오 API)
3. 위/경도로 Naver Maps 타일 이미지 생성
4. 마라톤 이미지와 타일에서 anchor 패치 생성
5. LoFTR로 anchor 패치 매칭 → 매칭 시각화 및 분석
6. ...


## 1) OCR
**마라톤 경로 이미지에서 텍스트를 검출하여 주요 랜드마크 후보를 추출하는 단계**

스크립트: [path-to-gpx/src/01_ocr.py](./src/01_ocr.py)

실행 예시
```
.\path-to-gpx\src\01_ocr.py --image . \path-to-gpx\data\images\110.jpg --output-dir .\path-to-gpx\output\ocr\ --lang korean --min-confidence 0.2
```

출력 위치
- [path-to-gpx/output/ocr](./output/01.ocr)

주요 산출물
- 이미지: 110_ocr_visualized.png
- JSON: 110_ocr_detections.json
- CSV: 110_ocr_detections.csv
- JSON: 110_ocr_rejected.json

#### 문제점
- OCR이 마라톤 이미지에서 텍스트를 제대로 검출하지 못함
- 무엇이 중요한 랜드마크 텍스트인지 판단이 어려움

#### 대체 방안:
- 하드코딩을 통해 주요 랜드마크 텍스트 후보를 미리 정의하여 검색
- 예시: '남산타워', '여의도한강공원', '빛가람대교' 등


## 2) Geocoding
**OCR로 검출된 텍스트를 기반으로 위/경도로 변환하는 단계 (카카오 로컬 API 사용)**

> 카카오 API를 사용한 이유는, OpenStreetMap Nominatim API가 한국 내 장소에 대한 지오코딩 정확도가 낮기 때문이다. 카카오 API는 한국 장소에 대한 지오코딩이 상대적으로 더 정확하므로, 마라톤 경로 이미지에서 검출된 텍스트를 효과적으로 위/경도로 변환할 수 있다.

스크립트: [path-to-gpx/src/02_geo_coding.py](./src/02_geo_coding.py)

실행 예시
```
python .\path-to-gpx\src\02_geo_coding.py --ocr-json .\path-to-gpx\output\01.ocr\110_ocr_detections.json --top-k 5
```

출력 위치
- [path-to-gpx/output/02.geo_coding](./output/02.geocoding/)

주요 산출물
- JSON: 110_geocode_candidates.json


## 3) Map Tiles 생성
**위/경도를 이용하여 Naver Maps Static API로 지도 타일 이미지 생성**

스크립트: [path-to-gpx/src/03_map_tiles.py](./src/03_map_tiles.py)

실행 예시
	python .\path-to-gpx\src\03_map_tiles.py --geocode-json .\path-to-gpx\output\02.geo_coding\110_geocode_candidates.json --zoom 14 --tile-size 700 --primary-keyword 여의도

출력 위치
- [path-to-gpx/output/03_map_tiles](./output/03.map_tiles)
- [path-to-gpx/output/03_map_tiles/tiles](./output/03.map_tiles/tiles)

주요 산출물
- 타일 이미지(핵심 1장, 선택 좌표가 중심): 110_z14_x13959_y6488.png
- 리포트 JSON: 110_map_tiles_report.json


## 4) anchor 패치 생성
**마라톤 이미지에서 OCR로 검출된 랜드마크 주변 패치와, Naver Maps 타일에서 대응 패치를 생성하여 매칭 준비**

스크립트: [path-to-gpx/src/04_anchor_patches.py](./src/04_anchor_patches.py)

실행 예시
```
python .\path-to-gpx\src\04_anchor_patches.py \
	--ocr-json .\path-to-gpx\output\01.ocr\110_ocr_detections.json \
	--geocode-json .\path-to-gpx\output\02.geo_coding\110_geocode_candidates.json \
	--zoom 14 \
	--patch-size 512 \
	--output-dir .\path-to-gpx\output\04.anchor_patches\
```
출력 위치
- [path-to-gpx/output/04.anchor_patches](./output/04.anchors/)

주요 산출물
- JSON: 110_anchor_patches.json
- 패치 이미지: 110_anchor_0_source.png, 110_anchor_0_reference.png, ...


## 5) LoFTR 매칭 및 시각화
**전통적인 컴퓨터비전 특징점 매칭 알고리즘인 SIFT 대신, 딥러닝 기반 LoFTR 매칭을 사용하여 anchor 패치 간의 대응점을 찾고, 다양한 시각화 방식으로 매칭 결과를 분석**

스크립트: [path-to-gpx/src/05_loftr_match.py](./src/05_loftr_match.py)

실행 예시
```
python .\path-to-gpx\src\05_loftr_match.py \
	--anchors-json .\path-to-gpx\output\04.anchor_patches\110_anchor_patches.json \
	--confidence 0.2 \
	--magsac-thr 3.0 \
	--output-dir .\path-to-gpx\output\loftr_matches\ \
	--save-visualizations
```
출력 위치
- [path-to-gpx/output/loftr_matches](./output/05.loftr/)
- [path-to-gpx/output/loftr_matches/visualizations](./output/05.loftr/083_match_visualizations)

주요 산출물
- JSON: 110_loftr_matches.json
- 시각화 이미지: 02_all_matches.jpg, 03_grid_view.jpg, 04_sequential_combined.jpg, anchor_110_matches_0001.jpg, ...


