# 흐름 요약
1. OCR: 마라톤 이미지에서 텍스트 검출
2. Geocoding: OCR 텍스트 기반 좌표 결정
3. OSM: 결정 좌표 기준 OpenStreetMap 타일 저장
4. Homography: 마라톤 이미지와 실제 지도 특징점 매칭

## 1) OCR

스크립트: [path-to-gpx/src/01_ocr.py](path-to-gpx/src/01_ocr.py)

실행 예시
	python .\path-to-gpx\src\01_ocr.py --image .\path-to-gpx\data\images\110.jpg --output-dir .\path-to-gpx\output\ocr\ --lang korean --min-confidence 0.2

출력 위치
- [path-to-gpx/output/ocr](path-to-gpx/output/ocr)

주요 산출물
- 이미지: 110_ocr_visualized.png
- JSON: 110_ocr_detections.json
- CSV: 110_ocr_detections.csv
- JSON: 110_ocr_rejected.json

## 2) Geocoding

스크립트: [path-to-gpx/src/02_geo_coding.py](path-to-gpx/src/02_geo_coding.py)

실행 예시
	python .\path-to-gpx\src\02_geo_coding.py --ocr-json .\path-to-gpx\output\ocr\110_ocr_detections.json --top-k 5

출력 위치
- [path-to-gpx/output/geocoding](path-to-gpx/output/geocoding)

주요 산출물
- JSON: 110_geocode_candidates.json

## 3) OSM Tile 생성

스크립트: [path-to-gpx/src/03_osm.py](path-to-gpx/src/03_osm.py)

실행 예시
	python .\path-to-gpx\src\03_osm.py --geocode-json .\path-to-gpx\output\geocoding\110_geocode_candidates.json --zoom-levels 12,14,16

출력 위치
- [path-to-gpx/output/osm](path-to-gpx/output/osm)
- [path-to-gpx/output/osm/tiles](path-to-gpx/output/osm/tiles)

주요 산출물
- 타일 이미지: 110_z12_x3489_y1622.png
- 타일 이미지: 110_z14_x13959_y6488.png
- 타일 이미지: 110_z16_x55836_y25955.png
- 리포트 JSON: 110_osm_tiles_report.json

## 4) Homography 특징점 매칭

스크립트: [path-to-gpx/src/04_homography.py](path-to-gpx/src/04_homography.py)

실행 예시
	python .\path-to-gpx\src\04_homography.py --marathon-image .\path-to-gpx\data\images\110.jpg --map-image .\path-to-gpx\output\osm\tiles\110_z14_x13959_y6488.png

출력 위치
- [path-to-gpx/output/homography](path-to-gpx/output/homography)

주요 산출물
- 매칭 시각화: 110_to_110_z14_x13959_y6488_matches.png
- 인라이어 시각화: 110_to_110_z14_x13959_y6488_inliers.png
- 오버레이: 110_to_110_z14_x13959_y6488_overlay.png
- 리포트 JSON: 110_to_110_z14_x13959_y6488_report.json

# 지금 문제점
뽑아낸 타일이 실제 마라톤 이미지와 잘 매칭이 안됨

이유는?

# 만약 특징점이 잘 매칭이 된다면 그 다음은?
- 매칭된 특징점간의 호모그래피 계산
- 마라톤 이미지와 타일 이미지 간의 투영 변환
- 투영 행렬을 구하면 마라톤 이미지의 좌표를 지도 좌표로 변환 가능
- 변환된 좌표를 GPX 포맷으로 저장하여 GPS 트랙 생성