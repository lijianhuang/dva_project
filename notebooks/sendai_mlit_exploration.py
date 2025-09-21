#!/usr/bin/env python
# coding: utf-8

# # Sendai MLIT Land Data Exploration
# This notebook pulls sample land transaction records for Sendai from the MLIT Web Land API and builds
# lightweight metrics you can reuse in the course project. It is written at an intermediate Python level
# and only depends on common data packages (requests, pandas, plotly).
# 

# ## Prerequisites
# 1. Store your API token in an environment variable named `MLIT_API_KEY` (for example, in `.env`).
# 2. Install dependencies if needed: `pip install pandas requests plotly python-dotenv` (python-dotenv is optional).
# 3. Run the cells in order; they fetch a few hundred records so execution stays quick.
# 

# In[1]:


import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import plotly.express as px

pd.options.display.max_columns = 30
pd.options.display.float_format = '{:,.0f}'.format

def load_env_variable(name: str, *, env_path: str = '.env') -> str:
    """Return an environment variable, optionally reading a local .env file."""
    value = os.getenv(name)
    if value:
        return value
    env_file = Path(env_path)
    if env_file.exists():
        for line in env_file.read_text(encoding='utf-8').splitlines():
            if not line or line.strip().startswith('#'):
                continue
            if '=' not in line:
                continue
            key, raw_val = line.split('=', 1)
            if key.strip() == name:
                return raw_val.strip().replace(chr(34), '').replace("'", '')
    raise RuntimeError(f'Environment variable {name} is not set. Set it before running the notebook.')

MLIT_API_KEY = load_env_variable('MLIT_API_KEY')
MLIT_BASE_URL = 'https://www.reinfolib.mlit.go.jp/ex-api/external'

# Endpoint identifiers from the API manual
MLIT_TRADE_ENDPOINT = 'XIT001'           # Real estate transaction price information
MLIT_MUNICIPALITIES_ENDPOINT = 'XIT002'  # Prefecture municipality list
MLIT_APPRAISAL_ENDPOINT = 'XCT001'       # Appraisal report information
MLIT_LAND_PRICE_TILE_ENDPOINT = 'XPT002' # Land price point tiles (GeoJSON / vector tiles)


# In[2]:


def call_mlit_api(endpoint: str, params: Dict[str, str]) -> Dict:
    """Call MLIT Real Estate Information Library API endpoint and return the parsed JSON body."""
    url = f"{MLIT_BASE_URL}/{endpoint}"
    print(f'Making API call to: {url}')
    print(f'With parameters: {params}')

    response = requests.get(
        url,
        params=params,
        headers={'Ocp-Apim-Subscription-Key': MLIT_API_KEY},
        timeout=30
    )

    print(f'Response status: {response.status_code}')

    if response.status_code in (204, 404):
        print(f'No records returned for this call (HTTP {response.status_code}).')
        return {'status': 'OK', 'data': []}

    if response.status_code != 200:
        preview = response.text[:500]
        print(f'API Error Response Preview: {preview}')
        response.raise_for_status()

    try:
        payload = response.json()
    except ValueError as exc:
        print('Failed to decode JSON; first bytes:', response.content[:200])
        raise exc

    api_status = payload.get('status')
    if api_status and api_status.upper() != 'OK':
        raise RuntimeError(f"API returned status {api_status}: {payload}")

    data = payload.get('data')
    print(f"Records returned: {len(data) if isinstance(data, list) else 'n/a'}")
    return payload

def fetch_trade_data(
    pref_code: Optional[str] = None,
    *,
    city_code: Optional[str] = None,
    station_code: Optional[str] = None,
    price_classification: Optional[str] = '01',
    start_year: int = 2023,
    start_quarter: int = 1,
    end_year: Optional[int] = None,
    end_quarter: Optional[int] = None,
    language: str = 'en'
) -> List[Dict]:
    """Fetch trade records for a prefecture, city, or station across a year/quarter span."""
    if not any([pref_code, city_code, station_code]):
        raise ValueError('At least one of pref_code, city_code, or station_code must be provided.')

    if end_year is None:
        end_year = start_year

    if end_quarter is None:
        end_quarter = start_quarter if end_year == start_year else 4

    if end_year < start_year:
        raise ValueError('end_year must be greater than or equal to start_year.')

    for label, value in [('start_quarter', start_quarter), ('end_quarter', end_quarter)]:
        if not 1 <= value <= 4:
            raise ValueError(f'{label} must be between 1 and 4.')

    if start_year == end_year and end_quarter < start_quarter:
        raise ValueError('When start_year == end_year, end_quarter must be greater than or equal to start_quarter.')

    records: List[Dict] = []

    for year in range(start_year, end_year + 1):
        quarter_start = start_quarter if year == start_year else 1
        quarter_end = end_quarter if year == end_year else 4

        for quarter in range(quarter_start, quarter_end + 1):
            params: Dict[str, str] = {
                'year': str(year),
                'language': language,
                'quarter': str(quarter)
            }

            if price_classification:
                params['priceClassification'] = price_classification
            if pref_code:
                params['area'] = str(pref_code)
            if city_code:
                params['city'] = str(city_code)
            if station_code:
                params['station'] = str(station_code)

            print(f'Fetching data for {year} Q{quarter}...')
            payload = call_mlit_api(MLIT_TRADE_ENDPOINT, params)
            batch = payload.get('data') or []
            print(f'Added {len(batch)} records for {year} Q{quarter}.')

            for record in batch:
                record.setdefault('Year', str(year))
                record.setdefault('Quarter', str(quarter))
                if price_classification:
                    record.setdefault('PriceClassification', price_classification)

            records.extend(batch)

    print(f'Total records fetched: {len(records)}')
    return records

def fetch_municipalities(
    pref_code: str,
    *,
    language: str = 'en'
) -> List[Dict]:
    """Return the list of municipalities for a prefecture via MLIT XIT002."""
    params: Dict[str, str] = {'area': str(pref_code)}
    if language:
        params['language'] = language

    payload = call_mlit_api(MLIT_MUNICIPALITIES_ENDPOINT, params)
    records = payload.get('data') or []
    print(f'Returned {len(records)} municipalities for prefecture {pref_code}.')
    return records

def fetch_appraisal_records(
    year: int,
    pref_code: str,
    *,
    division: str = '00',
    language: str = 'en'
) -> List[Dict]:
    """Return appraisal report information for a prefecture/year using MLIT XCT001."""
    params: Dict[str, str] = {
        'year': str(year),
        'area': str(pref_code),
        'division': str(division)
    }
    if language:
        params['language'] = language

    payload = call_mlit_api(MLIT_APPRAISAL_ENDPOINT, params)
    records = payload.get('data') or []
    print(f'Returned {len(records)} appraisal records for {year} prefecture {pref_code} division {division}.')
    return records

def slippy_tile_index(lon: float, lat: float, zoom: int) -> Tuple[int, int]:
    """Convert lon/lat coordinates to Web Mercator slippy tile indices."""
    lat_rad = math.radians(lat)
    n = 2 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile

def slippy_tile_bounds(x: int, y: int, zoom: int) -> Tuple[float, float, float, float]:
    """Return (lon_min, lat_min, lon_max, lat_max) for a Web Mercator slippy tile."""
    n = 2 ** zoom
    lon_min = x / n * 360.0 - 180.0
    lon_max = (x + 1) / n * 360.0 - 180.0
    lat_min = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    lat_max = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lon_min, lat_min, lon_max, lat_max

def geojson_features_to_dataframe(geojson: Dict) -> pd.DataFrame:
    """Convert a GeoJSON FeatureCollection to a pandas DataFrame with lat/lon columns."""
    features = geojson.get('features') or []
    rows: List[Dict] = []
    for feature in features:
        props = feature.get('properties', {}).copy()
        geometry = feature.get('geometry') or {}
        if geometry.get('type') == 'Point':
            coords = geometry.get('coordinates', [None, None])
            props['longitude'] = coords[0]
            props['latitude'] = coords[1]
        rows.append(props)
    return pd.DataFrame(rows)

def collect_land_price_features(
    *,
    center_lat: float,
    center_lon: float,
    zoom: int,
    year: int,
    tile_radius: int = 0,
    price_classification: Optional[str] = None,
    use_category_codes: Optional[str] = None,
    response_format: str = 'geojson'
) -> Dict:
    """Collect land price point features (XPT002) around a center coordinate."""
    x_center, y_center = slippy_tile_index(center_lon, center_lat, zoom)
    all_features: List[Dict] = []
    tile_summaries: List[Dict] = []

    for dx in range(-tile_radius, tile_radius + 1):
        for dy in range(-tile_radius, tile_radius + 1):
            x = x_center + dx
            y = y_center + dy
            params: Dict[str, str] = {
                'response_format': response_format,
                'z': str(zoom),
                'x': str(x),
                'y': str(y),
                'year': str(year)
            }
            if price_classification:
                params['priceClassification'] = price_classification
            if use_category_codes:
                params['useCategoryCode'] = use_category_codes

            payload = call_mlit_api(MLIT_LAND_PRICE_TILE_ENDPOINT, params)
            features = payload.get('features') or []
            print(f'Tile z={zoom} x={x} y={y}: {len(features)} features')
            all_features.extend(features)
            lon_min, lat_min, lon_max, lat_max = slippy_tile_bounds(x, y, zoom)
            tile_summaries.append({
                'zoom': zoom,
                'x': x,
                'y': y,
                'feature_count': len(features),
                'lon_min': lon_min,
                'lat_min': lat_min,
                'lon_max': lon_max,
                'lat_max': lat_max
            })

    geojson = {'type': 'FeatureCollection', 'features': all_features}
    dataframe = geojson_features_to_dataframe(geojson)
    return {'geojson': geojson, 'dataframe': dataframe, 'tiles': tile_summaries}


def parse_numeric_value(value: Any) -> Optional[float]:
    """Return a float from MLIT numeric fields that include trailing units."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace("−", "-")
        if not cleaned:
            return None
        match = re.search(r"-?\d+(?:,\d{3})*(?:\.\d+)?", cleaned)
        if not match:
            return None
        numeric_text = match.group(0).replace(",", "")
        try:
            return float(numeric_text)
        except ValueError:
            return None
    return None


# ## Fetch transactions for Sendai
# We start with Miyagi Prefecture (pref=04) and the Sendai municipal code (city=04100).
# Adjust the quarters to explore longer histories or different seasonal windows.
# 

# In[3]:


# Configure the time span and filters for the MLIT fetch
FETCH_PREF_CODE = '04'  # Miyagi Prefecture
FETCH_CITY_CODE = None  # Provide a 5-digit municipality code or None
FETCH_STATION_CODE = None  # Provide a 6-digit station code or None
FETCH_PRICE_CLASSIFICATION = '01'  # '01' transactions, '02' contract prices, None for both
FETCH_LANGUAGE = 'en'
FETCH_START_YEAR = 2007
FETCH_START_QUARTER = 1
FETCH_END_YEAR = 2025  # Set to None to default to FETCH_START_YEAR
FETCH_END_QUARTER = 4  # Set to None to default (4 when spanning multiple years)
APPLY_SENDAI_FILTER = True  # Keep only Sendai ward records when Miyagi data is fetched

# Additional endpoint configuration
APPRAISAL_DIVISION = '00'      # 00: Residential land, see manual for all options
APPRAISAL_YEAR_OVERRIDE = None  # Set to an integer year to override the automatic selection

print('Testing MLIT XIT001 endpoint with sample parameters...')

sample_year = FETCH_END_YEAR or FETCH_START_YEAR
if sample_year != FETCH_START_YEAR and FETCH_END_QUARTER is None:
    sample_quarter = 4
else:
    sample_quarter = FETCH_END_QUARTER or FETCH_START_QUARTER

sample_params = {
    'area': str(FETCH_PREF_CODE),
    'year': str(sample_year),
    'quarter': str(sample_quarter),
    'language': FETCH_LANGUAGE,
}
if FETCH_PRICE_CLASSIFICATION:
    sample_params['priceClassification'] = FETCH_PRICE_CLASSIFICATION
if FETCH_CITY_CODE:
    sample_params['city'] = str(FETCH_CITY_CODE)
if FETCH_STATION_CODE:
    sample_params['station'] = str(FETCH_STATION_CODE)

sample_payload = call_mlit_api(MLIT_TRADE_ENDPOINT, sample_params)
sample_data = sample_payload.get('data') or []
print(f'Sample request returned {len(sample_data)} records.')
if sample_data:
    print('First sample record:')
    print(json.dumps(sample_data[0], ensure_ascii=False, indent=2)[:800])

# Known ward codes for Sendai City (Aoba, Miyagino, Wakabayashi, Taihaku, Izumi)
SENDAI_WARD_CODES = {'4101', '4102', '4103', '4104', '4105'}

# Prepare cache paths
period_end_year = FETCH_END_YEAR or FETCH_START_YEAR
if period_end_year != FETCH_START_YEAR and FETCH_END_QUARTER is None:
    period_end_quarter = 4
else:
    period_end_quarter = FETCH_END_QUARTER or FETCH_START_QUARTER

period_label = f"{FETCH_START_YEAR}Q{FETCH_START_QUARTER}_to_{period_end_year}Q{period_end_quarter}"

APPRAISAL_YEAR = APPRAISAL_YEAR_OVERRIDE or period_end_year

data_raw_dir = Path('data/raw')
data_raw_dir.mkdir(parents=True, exist_ok=True)
reference_dir = Path('data/reference')
reference_dir.mkdir(parents=True, exist_ok=True)

trade_cache_path = data_raw_dir / f'trades_{FETCH_PREF_CODE}_{period_label}.json'
appraisal_cache_path = data_raw_dir / f'appraisal_{FETCH_PREF_CODE}_{APPRAISAL_DIVISION}_{APPRAISAL_YEAR}.json'
municipalities_cache_path = reference_dir / f'municipalities_{FETCH_PREF_CODE}.json'
sample_path = Path('data/sample_sendai_transactions.json')

def load_local_records(path: Path, label: str) -> List[Dict]:
    print(f'Loading {label} data from {path}.')
    return json.loads(path.read_text(encoding='utf-8'))

print(f"Attempting to fetch data from {FETCH_START_YEAR} Q{FETCH_START_QUARTER} through {period_end_year} Q{period_end_quarter}...")

raw_records: List[Dict] = []
try:
    raw_records = fetch_trade_data(
        FETCH_PREF_CODE,
        city_code=FETCH_CITY_CODE,
        station_code=FETCH_STATION_CODE,
        price_classification=FETCH_PRICE_CLASSIFICATION,
        start_year=FETCH_START_YEAR,
        start_quarter=FETCH_START_QUARTER,
        end_year=FETCH_END_YEAR,
        end_quarter=FETCH_END_QUARTER,
        language=FETCH_LANGUAGE
    )
    if raw_records:
        trade_cache_path.write_text(json.dumps(raw_records, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'Fetched {len(raw_records)} records from API and cached to {trade_cache_path}.')
    else:
        print('API returned no records; attempting to use cached or sample data.')
except requests.exceptions.RequestException as exc:
    print(f'API call failed due to network issue: {exc}')
except Exception as exc:
    print(f'API call failed with an unexpected error: {exc}')

if not raw_records:
    if trade_cache_path.exists():
        raw_records = load_local_records(trade_cache_path, 'cached trade API')
    elif sample_path.exists():
        raw_records = load_local_records(sample_path, 'sample')
    else:
        raise SystemExit('No API data available. Check your network connection or add sample data at data/sample_sendai_transactions.json.')

transactions_df = pd.DataFrame(raw_records)
print(f'Loaded {len(transactions_df)} transaction records into DataFrame.')

if APPLY_SENDAI_FILTER and not transactions_df.empty:
    if 'MunicipalityCode' in transactions_df.columns:
        sendai_df = transactions_df[transactions_df['MunicipalityCode'].isin(SENDAI_WARD_CODES)].copy()
        if sendai_df.empty:
            print('Sendai-specific filter returned 0 rows; retaining prefecture-wide data.')
        else:
            transactions_df = sendai_df.reset_index(drop=True)
            print(f'Filtered to {len(transactions_df)} Sendai ward records.')
    else:
        print('MunicipalityCode column missing; skipping Sendai filter.')
elif transactions_df.empty:
    print('No records loaded; downstream analysis will need to rely on fallback data.')
else:
    print('Sendai filter disabled; keeping full dataset.')

transactions_df.head()


# In[4]:


# Fetch municipality reference data for the configured prefecture
municipality_records: List[Dict] = []
try:
    municipality_records = fetch_municipalities(FETCH_PREF_CODE, language=FETCH_LANGUAGE)
    if municipality_records:
        municipalities_cache_path.write_text(json.dumps(municipality_records, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'Cached municipality list to {municipalities_cache_path}.')
except requests.exceptions.RequestException as exc:
    print(f'Municipality lookup failed due to network issue: {exc}')
except Exception as exc:
    print(f'Municipality lookup failed with an unexpected error: {exc}')

if not municipality_records and municipalities_cache_path.exists():
    municipality_records = load_local_records(municipalities_cache_path, 'cached municipality list')

if not municipality_records:
    raise SystemExit('Unable to load municipality metadata for the configured prefecture.')

municipalities_df = pd.DataFrame(municipality_records)
print(f'Retrieved {len(municipalities_df)} municipalities for prefecture {FETCH_PREF_CODE}.')
municipalities_df.head()


# In[5]:


# Fetch appraisal report information for the configured prefecture / division
appraisal_records: List[Dict] = []
try:
    appraisal_records = fetch_appraisal_records(
        APPRAISAL_YEAR,
        FETCH_PREF_CODE,
        division=APPRAISAL_DIVISION,
        language=FETCH_LANGUAGE
    )
    if appraisal_records:
        appraisal_cache_path.write_text(json.dumps(appraisal_records, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'Cached appraisal data to {appraisal_cache_path}.')
except requests.exceptions.RequestException as exc:
    print(f'Appraisal lookup failed due to network issue: {exc}')
except Exception as exc:
    print(f'Appraisal lookup failed with an unexpected error: {exc}')

if not appraisal_records and appraisal_cache_path.exists():
    appraisal_records = load_local_records(appraisal_cache_path, 'cached appraisal data')

if not appraisal_records:
    print('No appraisal records available for the requested parameters.')
    appraisal_df = pd.DataFrame()
    appraisal_preview_df = appraisal_df
else:
    appraisal_df = pd.DataFrame(appraisal_records)
    numeric_cols = [col for col in ['1㎡当たりの価格', '標準地 交通施設の状況 距離'] if col in appraisal_df.columns]
    for col in numeric_cols:
        appraisal_df[col] = pd.to_numeric(appraisal_df[col], errors='coerce')

    column_map = {
        '価格時点': 'appraisal_date',
        '標準地番号 地域名': 'district_name',
        '標準地番号 用途区分': 'use_category_code',
        '1㎡当たりの価格': 'price_per_sqm',
        '標準地 交通施設の状況 交通施設': 'nearest_transport',
        '標準地 交通施設の状況 距離': 'transport_distance_m',
        '緯度': 'latitude',
        '経度': 'longitude'
    }
    appraisal_df.rename(columns={k: v for k, v in column_map.items() if k in appraisal_df.columns}, inplace=True)
    summary_columns = [col for col in column_map.values() if col in appraisal_df.columns]
    appraisal_preview_df = appraisal_df[summary_columns].head() if summary_columns else appraisal_df.head()

print(f'Appraisal record count: {len(appraisal_df)}')
appraisal_preview_df


# ## Land price point tiles (XPT002)

# In[6]:


# Collect land price point features (XPT002) around central Sendai
LAND_PRICE_CENTER = {'lat': 38.2682, 'lon': 140.8694}
LAND_PRICE_TILE_ZOOM = 13
LAND_PRICE_TILE_RADIUS = 1  # grabs a (2 * radius + 1)^2 grid of tiles
LAND_PRICE_YEAR_START = max(APPRAISAL_YEAR - 2, 2011)
LAND_PRICE_YEAR_END = APPRAISAL_YEAR
LAND_PRICE_MANUAL_YEARS = None  # override with a list, e.g., [2021, 2023]
LAND_PRICE_USE_CODES = '00,03,05'  # residential / commercial focus areas
LAND_PRICE_PRICE_CLASSIFICATION = None  # optional: filter by price classification code

if LAND_PRICE_MANUAL_YEARS:
    LAND_PRICE_YEARS = sorted({int(year) for year in LAND_PRICE_MANUAL_YEARS})
else:
    LAND_PRICE_YEARS = list(range(int(LAND_PRICE_YEAR_START), int(LAND_PRICE_YEAR_END) + 1))

land_price_frames = []
tile_summaries = []

for year in LAND_PRICE_YEARS:
    fetch_result = collect_land_price_features(
        center_lat=LAND_PRICE_CENTER['lat'],
        center_lon=LAND_PRICE_CENTER['lon'],
        zoom=LAND_PRICE_TILE_ZOOM,
        year=int(year),
        tile_radius=LAND_PRICE_TILE_RADIUS,
        price_classification=LAND_PRICE_PRICE_CLASSIFICATION,
        use_category_codes=LAND_PRICE_USE_CODES
    )
    frame = fetch_result['dataframe'].copy()
    frame['year'] = int(year)
    land_price_frames.append(frame)
    for tile in fetch_result['tiles']:
        tile_rec = dict(tile)
        tile_rec['year'] = int(year)
        tile_summaries.append(tile_rec)
    if fetch_result['geojson']['features']:
        geojson_path = data_raw_dir / f'land_price_{FETCH_PREF_CODE}_{year}_z{LAND_PRICE_TILE_ZOOM}_r{LAND_PRICE_TILE_RADIUS}.geojson'
        geojson_path.write_text(json.dumps(fetch_result['geojson'], ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'Stored land price GeoJSON for {year} at {geojson_path}.')
    else:
        print(f'No land price features returned for year {year}.')

land_price_df = (
    pd.concat(land_price_frames, ignore_index=True)
    if land_price_frames else pd.DataFrame()
)
land_price_tiles_df = pd.DataFrame(tile_summaries)

value_aliases = {
    'u_current_years_price_ja': 'current_price',
    'last_years_price': 'last_year_price',
    'u_cadastral_ja': 'parcel_area_sqm',
    'u_regulations_building_coverage_ratio_ja': 'coverage_ratio_pct',
    'u_regulations_floor_area_ratio_ja': 'floor_area_ratio_pct'
}
for source, alias in value_aliases.items():
    if source in land_price_df.columns:
        land_price_df[alias] = land_price_df[source].apply(parse_numeric_value)

if {'current_price', 'parcel_area_sqm'}.issubset(land_price_df.columns):
    land_price_df['price_per_sqm'] = land_price_df['current_price']
    land_price_df['estimated_site_price'] = land_price_df['current_price'] * land_price_df['parcel_area_sqm']

land_price_categories = (
    sorted(land_price_df['use_category_name_ja'].dropna().unique().tolist())
    if not land_price_df.empty else []
)
print(f'Total land price points collected: {len(land_price_df)} across {land_price_df["year"].nunique() if not land_price_df.empty else 0} year(s).')


# In[7]:


# Summarise the tiles that were fetched
if 'land_price_tiles_df' not in locals() or land_price_tiles_df.empty:
    print('Tile summary is empty.')
else:
    tile_summary_df = land_price_tiles_df[['year', 'zoom', 'x', 'y', 'feature_count', 'lon_min', 'lat_min', 'lon_max', 'lat_max']].copy()
    tile_summary_df.sort_values(['year', 'feature_count'], ascending=[True, False], inplace=True)
    tile_summary_df['lon_range'] = tile_summary_df.apply(lambda row: f"{row['lon_min']:.3f} to {row['lon_max']:.3f}", axis=1)
    tile_summary_df['lat_range'] = tile_summary_df.apply(lambda row: f"{row['lat_min']:.3f} to {row['lat_max']:.3f}", axis=1)
    tile_summary_df[['year', 'zoom', 'x', 'y', 'feature_count', 'lon_range', 'lat_range']]


# In[8]:


# Quick look at land price records
if 'land_price_df' not in locals() or land_price_df.empty:
    print('No land price data collected.')
else:
    preview_columns = [
        'year',
        'use_category_name_ja',
        'nearest_station_name_ja',
        'current_price',
        'price_per_sqm',
        'parcel_area_sqm',
        'latitude',
        'longitude'
    ]
    preview_columns = [col for col in preview_columns if col in land_price_df.columns]
    land_price_df[preview_columns].head()


# ### Interactive Sendai map

# In[9]:


# Configure filters for the land price map
LAND_PRICE_INCLUDED_CATEGORIES = None  # e.g., ['住宅地', '商業地']
LAND_PRICE_YEAR_RANGE = (None, None)   # (start_year, end_year)
LAND_PRICE_PRICE_RANGE = (None, None)  # (min_price, max_price)
LAND_PRICE_SIZE_RANGE = (None, None)   # (min_sqm, max_sqm)
LAND_PRICE_COLOR_FIELD = 'current_price'
LAND_PRICE_SIZE_FIELD = 'parcel_area_sqm'
MAPBOX_STYLE = 'carto-positron'


# In[10]:


# Plot land price points on a tile-backed map of Sendai
if 'land_price_df' not in locals() or land_price_df.empty:
    raise SystemExit('Land price DataFrame not available. Run the land price tile cell above first.')

map_df = land_price_df.copy()
if LAND_PRICE_INCLUDED_CATEGORIES:
    map_df = map_df[map_df['use_category_name_ja'].isin(LAND_PRICE_INCLUDED_CATEGORIES)]

year_min, year_max = LAND_PRICE_YEAR_RANGE
if year_min is not None:
    map_df = map_df[map_df['year'] >= year_min]
if year_max is not None:
    map_df = map_df[map_df['year'] <= year_max]

price_min, price_max = LAND_PRICE_PRICE_RANGE
if price_min is not None and 'current_price' in map_df.columns:
    map_df = map_df[map_df['current_price'] >= price_min]
if price_max is not None and 'current_price' in map_df.columns:
    map_df = map_df[map_df['current_price'] <= price_max]

size_min, size_max = LAND_PRICE_SIZE_RANGE
if size_min is not None and 'parcel_area_sqm' in map_df.columns:
    map_df = map_df[map_df['parcel_area_sqm'] >= size_min]
if size_max is not None and 'parcel_area_sqm' in map_df.columns:
    map_df = map_df[map_df['parcel_area_sqm'] <= size_max]

color_field = LAND_PRICE_COLOR_FIELD if LAND_PRICE_COLOR_FIELD in map_df.columns else None
size_field = LAND_PRICE_SIZE_FIELD if LAND_PRICE_SIZE_FIELD in map_df.columns else None

numeric_fields = [field for field in (color_field, size_field) if field]
for field in numeric_fields:
    map_df[field] = pd.to_numeric(map_df[field], errors='coerce')

if numeric_fields:
    map_df = map_df.dropna(subset=numeric_fields)

if map_df.empty:
    raise SystemExit('No land price points remain after filtering.')

map_df = map_df.sort_values(color_field or 'year', ascending=False)

hover_data = {}
for col in ['use_category_name_ja', 'nearest_station_name_ja', 'price_per_sqm', 'parcel_area_sqm']:
    if col in map_df.columns and col != color_field:
        hover_data[col] = True
if color_field:
    hover_data[color_field] = ':.0f'

animation_frame = 'year' if map_df['year'].nunique() > 1 else None
map_center = {'lat': map_df['latitude'].mean(), 'lon': map_df['longitude'].mean()}

fig_land_map = px.scatter_mapbox(
    map_df,
    lat='latitude',
    lon='longitude',
    color=color_field,
    size=size_field,
    hover_name='standard_lot_number_ja' if 'standard_lot_number_ja' in map_df.columns else None,
    hover_data=hover_data,
    animation_frame=animation_frame,
    mapbox_style=MAPBOX_STYLE,
    color_continuous_scale='Viridis',
    size_max=18,
    zoom=11,
    center=map_center,
    title='Sendai land price tiles'
)
fig_land_map.update_traces(marker={'opacity': 0.75, 'sizemin': 4})
fig_land_map.update_layout(height=520, legend_title_text=color_field or 'Legend')
fig_land_map


# In[11]:


# Plot land price points on a tile-backed map of Sendai
if 'land_price_df' not in locals() or land_price_df.empty:
    raise SystemExit('Land price DataFrame not available. Run the land price tile cell above first.')

map_df = land_price_df.copy()
if LAND_PRICE_INCLUDED_CATEGORIES:
    map_df = map_df[map_df['use_category_name_ja'].isin(LAND_PRICE_INCLUDED_CATEGORIES)]

year_min, year_max = LAND_PRICE_YEAR_RANGE
if year_min is not None:
    map_df = map_df[map_df['year'] >= year_min]
if year_max is not None:
    map_df = map_df[map_df['year'] <= year_max]

price_min, price_max = LAND_PRICE_PRICE_RANGE
if price_min is not None and 'current_price' in map_df.columns:
    map_df = map_df[map_df['current_price'] >= price_min]
if price_max is not None and 'current_price' in map_df.columns:
    map_df = map_df[map_df['current_price'] <= price_max]

size_min, size_max = LAND_PRICE_SIZE_RANGE
if size_min is not None and 'parcel_area_sqm' in map_df.columns:
    map_df = map_df[map_df['parcel_area_sqm'] >= size_min]
if size_max is not None and 'parcel_area_sqm' in map_df.columns:
    map_df = map_df[map_df['parcel_area_sqm'] <= size_max]

color_field = LAND_PRICE_COLOR_FIELD if LAND_PRICE_COLOR_FIELD in map_df.columns else None
size_field = LAND_PRICE_SIZE_FIELD if LAND_PRICE_SIZE_FIELD in map_df.columns else None

numeric_fields = [field for field in (color_field, size_field) if field]
for field in numeric_fields:
    map_df[field] = pd.to_numeric(map_df[field], errors='coerce')

if numeric_fields:
    map_df = map_df.dropna(subset=numeric_fields)

if map_df.empty:
    raise SystemExit('No land price points remain after filtering.')

map_df = map_df.sort_values(color_field or 'year', ascending=False)

hover_data = {}
for col in ['use_category_name_ja', 'nearest_station_name_ja', 'price_per_sqm', 'parcel_area_sqm']:
    if col in map_df.columns and col != color_field:
        hover_data[col] = True
if color_field:
    hover_data[color_field] = ':.0f'

animation_frame = 'year' if map_df['year'].nunique() > 1 else None
map_center = {'lat': map_df['latitude'].mean(), 'lon': map_df['longitude'].mean()}

fig_land_map = px.scatter_mapbox(
    map_df,
    lat='latitude',
    lon='longitude',
    color=color_field,
    size=size_field,
    hover_name='standard_lot_number_ja' if 'standard_lot_number_ja' in map_df.columns else None,
    hover_data=hover_data,
    animation_frame=animation_frame,
    mapbox_style=MAPBOX_STYLE,
    color_continuous_scale='Viridis',
    size_max=18,
    zoom=11,
    center=map_center,
    title='Sendai land price tiles'
)
fig_land_map.update_traces(marker={'opacity': 0.75, 'sizemin': 4})
fig_land_map.update_layout(height=520, legend_title_text=color_field or 'Legend')
fig_land_map


# In[12]:


# Plot land price points on a tile-backed map of Sendai
if 'land_price_df' not in locals() or land_price_df.empty:
    raise SystemExit('Land price DataFrame not available. Run the land price tile cell above first.')

map_df = land_price_df.copy()
if LAND_PRICE_INCLUDED_CATEGORIES:
    map_df = map_df[map_df['use_category_name_ja'].isin(LAND_PRICE_INCLUDED_CATEGORIES)]

year_min, year_max = LAND_PRICE_YEAR_RANGE
if year_min is not None:
    map_df = map_df[map_df['year'] >= year_min]
if year_max is not None:
    map_df = map_df[map_df['year'] <= year_max]

price_min, price_max = LAND_PRICE_PRICE_RANGE
if price_min is not None and 'current_price' in map_df.columns:
    map_df = map_df[map_df['current_price'] >= price_min]
if price_max is not None and 'current_price' in map_df.columns:
    map_df = map_df[map_df['current_price'] <= price_max]

size_min, size_max = LAND_PRICE_SIZE_RANGE
if size_min is not None and 'parcel_area_sqm' in map_df.columns:
    map_df = map_df[map_df['parcel_area_sqm'] >= size_min]
if size_max is not None and 'parcel_area_sqm' in map_df.columns:
    map_df = map_df[map_df['parcel_area_sqm'] <= size_max]

color_field = LAND_PRICE_COLOR_FIELD if LAND_PRICE_COLOR_FIELD in map_df.columns else None
size_field = LAND_PRICE_SIZE_FIELD if LAND_PRICE_SIZE_FIELD in map_df.columns else None

numeric_fields = [field for field in (color_field, size_field) if field]
for field in numeric_fields:
    map_df[field] = pd.to_numeric(map_df[field], errors='coerce')

if numeric_fields:
    map_df = map_df.dropna(subset=numeric_fields)

if map_df.empty:
    raise SystemExit('No land price points remain after filtering.')

map_df = map_df.sort_values(color_field or 'year', ascending=False)

hover_data = {}
for col in ['use_category_name_ja', 'nearest_station_name_ja', 'price_per_sqm', 'parcel_area_sqm']:
    if col in map_df.columns and col != color_field:
        hover_data[col] = True
if color_field:
    hover_data[color_field] = ':.0f'

animation_frame = 'year' if map_df['year'].nunique() > 1 else None
map_center = {'lat': map_df['latitude'].mean(), 'lon': map_df['longitude'].mean()}

fig_land_map = px.scatter_mapbox(
    map_df,
    lat='latitude',
    lon='longitude',
    color=color_field,
    size=size_field,
    hover_name='standard_lot_number_ja' if 'standard_lot_number_ja' in map_df.columns else None,
    hover_data=hover_data,
    animation_frame=animation_frame,
    mapbox_style=MAPBOX_STYLE,
    color_continuous_scale='Viridis',
    size_max=18,
    zoom=11,
    center=map_center,
    title='Sendai land price tiles'
)
fig_land_map.update_traces(marker={'opacity': 0.75, 'sizemin': 4})
fig_land_map.update_layout(height=520, legend_title_text=color_field or 'Legend')
fig_land_map


# In[13]:


# Plot land price points on a tile-backed map of Sendai
if 'land_price_df' not in locals() or land_price_df.empty:
    raise SystemExit('Land price DataFrame not available. Run the land price tile cell above first.')

map_df = land_price_df.copy()
if LAND_PRICE_INCLUDED_CATEGORIES:
    map_df = map_df[map_df['use_category_name_ja'].isin(LAND_PRICE_INCLUDED_CATEGORIES)]

year_min, year_max = LAND_PRICE_YEAR_RANGE
if year_min is not None:
    map_df = map_df[map_df['year'] >= year_min]
if year_max is not None:
    map_df = map_df[map_df['year'] <= year_max]

price_min, price_max = LAND_PRICE_PRICE_RANGE
if price_min is not None and 'current_price' in map_df.columns:
    map_df = map_df[map_df['current_price'] >= price_min]
if price_max is not None and 'current_price' in map_df.columns:
    map_df = map_df[map_df['current_price'] <= price_max]

size_min, size_max = LAND_PRICE_SIZE_RANGE
if size_min is not None and 'parcel_area_sqm' in map_df.columns:
    map_df = map_df[map_df['parcel_area_sqm'] >= size_min]
if size_max is not None and 'parcel_area_sqm' in map_df.columns:
    map_df = map_df[map_df['parcel_area_sqm'] <= size_max]

color_field = LAND_PRICE_COLOR_FIELD if LAND_PRICE_COLOR_FIELD in map_df.columns else None
size_field = LAND_PRICE_SIZE_FIELD if LAND_PRICE_SIZE_FIELD in map_df.columns else None

numeric_fields = [field for field in (color_field, size_field) if field]
for field in numeric_fields:
    map_df[field] = pd.to_numeric(map_df[field], errors='coerce')

if numeric_fields:
    map_df = map_df.dropna(subset=numeric_fields)

if map_df.empty:
    raise SystemExit('No land price points remain after filtering.')

map_df = map_df.sort_values(color_field or 'year', ascending=False)

hover_data = {}
for col in ['use_category_name_ja', 'nearest_station_name_ja', 'price_per_sqm', 'parcel_area_sqm']:
    if col in map_df.columns and col != color_field:
        hover_data[col] = True
if color_field:
    hover_data[color_field] = ':.0f'

animation_frame = 'year' if map_df['year'].nunique() > 1 else None
map_center = {'lat': map_df['latitude'].mean(), 'lon': map_df['longitude'].mean()}

fig_land_map = px.scatter_mapbox(
    map_df,
    lat='latitude',
    lon='longitude',
    color=color_field,
    size=size_field,
    hover_name='standard_lot_number_ja' if 'standard_lot_number_ja' in map_df.columns else None,
    hover_data=hover_data,
    animation_frame=animation_frame,
    mapbox_style=MAPBOX_STYLE,
    color_continuous_scale='Viridis',
    size_max=18,
    zoom=11,
    center=map_center,
    title='Sendai land price tiles'
)
fig_land_map.update_traces(marker={'opacity': 0.75, 'sizemin': 4})
fig_land_map.update_layout(height=520, legend_title_text=color_field or 'Legend')
fig_land_map


# In[14]:


# Plot land price points on a tile-backed map of Sendai
if 'land_price_df' not in locals() or land_price_df.empty:
    raise SystemExit('Land price DataFrame not available. Run the land price tile cell above first.')

map_df = land_price_df.copy()
if LAND_PRICE_INCLUDED_CATEGORIES:
    map_df = map_df[map_df['use_category_name_ja'].isin(LAND_PRICE_INCLUDED_CATEGORIES)]

year_min, year_max = LAND_PRICE_YEAR_RANGE
if year_min is not None:
    map_df = map_df[map_df['year'] >= year_min]
if year_max is not None:
    map_df = map_df[map_df['year'] <= year_max]

price_min, price_max = LAND_PRICE_PRICE_RANGE
if price_min is not None and 'current_price' in map_df.columns:
    map_df = map_df[map_df['current_price'] >= price_min]
if price_max is not None and 'current_price' in map_df.columns:
    map_df = map_df[map_df['current_price'] <= price_max]

size_min, size_max = LAND_PRICE_SIZE_RANGE
if size_min is not None and 'parcel_area_sqm' in map_df.columns:
    map_df = map_df[map_df['parcel_area_sqm'] >= size_min]
if size_max is not None and 'parcel_area_sqm' in map_df.columns:
    map_df = map_df[map_df['parcel_area_sqm'] <= size_max]

color_field = LAND_PRICE_COLOR_FIELD if LAND_PRICE_COLOR_FIELD in map_df.columns else None
size_field = LAND_PRICE_SIZE_FIELD if LAND_PRICE_SIZE_FIELD in map_df.columns else None

numeric_fields = [field for field in (color_field, size_field) if field]
for field in numeric_fields:
    map_df[field] = pd.to_numeric(map_df[field], errors='coerce')

if numeric_fields:
    map_df = map_df.dropna(subset=numeric_fields)

if map_df.empty:
    raise SystemExit('No land price points remain after filtering.')

map_df = map_df.sort_values(color_field or 'year', ascending=False)

hover_data = {}
for col in ['use_category_name_ja', 'nearest_station_name_ja', 'price_per_sqm', 'parcel_area_sqm']:
    if col in map_df.columns and col != color_field:
        hover_data[col] = True
if color_field:
    hover_data[color_field] = ':.0f'

animation_frame = 'year' if map_df['year'].nunique() > 1 else None
map_center = {'lat': map_df['latitude'].mean(), 'lon': map_df['longitude'].mean()}

fig_land_map = px.scatter_mapbox(
    map_df,
    lat='latitude',
    lon='longitude',
    color=color_field,
    size=size_field,
    hover_name='standard_lot_number_ja' if 'standard_lot_number_ja' in map_df.columns else None,
    hover_data=hover_data,
    animation_frame=animation_frame,
    mapbox_style=MAPBOX_STYLE,
    color_continuous_scale='Viridis',
    size_max=18,
    zoom=11,
    center=map_center,
    title='Sendai land price tiles'
)
fig_land_map.update_traces(marker={'opacity': 0.75, 'sizemin': 4})
fig_land_map.update_layout(height=520, legend_title_text=color_field or 'Legend')
fig_land_map


# In[ ]:


# Summarise land price points by year and category
if 'land_price_df' not in locals() or land_price_df.empty:
    print('No land price data to summarise.')
else:
    summary_df = (
        land_price_df
        .groupby(['year', 'use_category_name_ja'], dropna=False)
        .agg(
            point_count=('year', 'size'),
            median_price=('current_price', 'median'),
            median_price_per_sqm=('price_per_sqm', 'median')
        )
        .reset_index()
        .sort_values(['year', 'point_count'], ascending=[True, False])
    )
    summary_df


# In[ ]:


sorted(transactions_df.columns.tolist())


# In[ ]:


if transactions_df.empty:
    raise SystemExit('No data downloaded. Check your API key, network connection, or parameters.')

numeric_columns = ['TradePrice', 'PricePerUnit', 'LandArea', 'FloorArea', 'TimeToNearestStation', 'DistanceToNearestStation']
for col in numeric_columns:
    if col in transactions_df.columns:
        transactions_df[col] = pd.to_numeric(transactions_df[col], errors='coerce')

year_col = 'DealYear' if 'DealYear' in transactions_df.columns else 'Year'
quarter_col = 'Quarter' if 'Quarter' in transactions_df.columns else None
if quarter_col and quarter_col in transactions_df.columns:
    transactions_df['deal_quarter'] = transactions_df[year_col].astype(int).astype(str) + ' Q' + transactions_df[quarter_col].astype(int).astype(str)
else:
    transactions_df['deal_quarter'] = transactions_df[year_col].astype(int).astype(str)

if {'TradePrice', 'LandArea'}.issubset(transactions_df.columns):
    transactions_df['price_per_sqm'] = transactions_df['TradePrice'] / transactions_df['LandArea']

display_columns = [col for col in ['Municipality', 'DistrictName', 'TradePrice', 'price_per_sqm', 'Purpose'] if col in transactions_df.columns]
transactions_df[display_columns].head()


# ## Metric 1: Median trade price per quarter
# A quick time-series shows whether prices are rising or falling within the selected period.
# 

# In[ ]:


price_summary = (
    transactions_df
    .dropna(subset=['TradePrice'])
    .groupby('deal_quarter', as_index=False)['TradePrice']
    .median()
)
price_summary.rename(columns={'TradePrice': 'median_trade_price'}, inplace=True)
price_summary


# In[ ]:


fig_prices = px.line(
    price_summary,
    x='deal_quarter',
    y='median_trade_price',
    markers=True,
    title='Median trade price per quarter (Sendai)'
)
fig_prices.update_layout(xaxis_title='Quarter', yaxis_title='JPY')
fig_prices


# ## Metric 2: Transaction counts by property type
# Grouping by the Type (or Purpose) column highlights which kinds of properties dominate activity.
# 

# In[ ]:


type_column = 'Type' if 'Type' in transactions_df.columns else 'Purpose' if 'Purpose' in transactions_df.columns else None
if type_column is None:
    raise SystemExit('Could not find a property-type column to group by.')

type_summary = (
    transactions_df
    .dropna(subset=[type_column])
    .groupby(['deal_quarter', type_column], as_index=False)
    .size()
)
type_summary.rename(columns={'size': 'transaction_count'}, inplace=True)
type_summary.head()


# In[ ]:


fig_types = px.bar(
    type_summary,
    x='deal_quarter',
    y='transaction_count',
    color=type_column,
    title='Transaction mix by property type',
    barmode='stack'
)
fig_types.update_layout(xaxis_title='Quarter', yaxis_title='Count')
fig_types


# ## Metric 3: Share of transactions near stations (optional)
# If DistanceToNearestStation is populated, we can classify which deals fall within a quick walk.
# 

# In[ ]:


if 'DistanceToNearestStation' in transactions_df.columns:
    def classify_distance(km: float) -> str:
        if km <= 0.5:
            return '<=0.5 km'
        if km <= 1.0:
            return '0.5-1.0 km'
        return '>1.0 km'

    station_bins = transactions_df['DistanceToNearestStation'].dropna().apply(classify_distance)
    transit_summary = station_bins.value_counts(normalize=True).rename_axis('distance_band').reset_index(name='share')
    transit_summary['share'] = transit_summary['share'] * 100
    fig_transit = px.bar(
        transit_summary,
        x='distance_band',
        y='share',
        title='Share of transactions by distance to nearest station'
    )
    fig_transit.update_layout(xaxis_title='Walk distance', yaxis_title='Percent of deals')
    fig_transit
else:
    print('Distance-to-station information is not available in this slice of the dataset.')


# ## Where to go next
# - Expand the time horizon or include other Miyagi municipalities to compare Sendai with nearby cities.
# - Join traffic census or GTFS stop data to enrich the accessibility story.
# - Export cleaned tables (for example, `transactions_df.to_parquet(...)`) for reuse in proposal and progress report deliverables.
# 
