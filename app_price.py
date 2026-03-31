import streamlit as st
import pandas as pd
import requests
from io import BytesIO, StringIO
from requests.auth import HTTPBasicAuth

st.set_page_config(page_title="Porównanie Price i Variants", layout="wide")

# ============================================================
# LOGOWANIE DO APLIKACJI
# ============================================================
APP_PASSWORD = st.secrets["app"]["app_password"]

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 Logowanie do aplikacji")
    st.markdown("Wpisz hasło, aby uzyskać dostęp do porównania danych.")
    with st.form("login_form"):
        password_input = st.text_input("Hasło:", type="password")
        submitted = st.form_submit_button("🔓 Zaloguj")
        if submitted:
            if password_input == APP_PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("❌ Nieprawidłowe hasło!")
    st.stop()
# ============================================================

st.title("Porównanie Price i Variants - Multi-sklepy")

HTTP_USERNAME = st.secrets["http_auth"]["username"]
HTTP_PASSWORD = st.secrets["http_auth"]["password"]

SHOP_TO_MPK = {
    '50stylepl': 'S501',
    'butysportowe': 'S503',
    'sizeerpl': 'S500',
    'sizeerde': 'G500',
    'sizeercz': 'CZ50',
    'sizeersk': 'SK50',
    'sizeerlt': 'LT50',
    'sizeerro': 'RO50',
    'timberland': 'S502',
    'jdsportspl': 'S512',
    'jdsportsro': 'RO55',
    'jdsportssk': 'SK52',
    'jdsportshu': 'HU52',
    'jdsportslt': 'LT52',
    'jdsportsbg': 'BG52',
    'jdsportscz': 'CZ55',
    'jdsportsua': 'UA52',
    'jdsportshr': 'HR52',
    'jdsportssi': 'SI52',
    'jdsportsee': 'EE52',
}

SHOP_DICT = {name: url for name, url in st.secrets["shop_urls"].items()}
MPK_TO_SHOP = {SHOP_TO_MPK.get(shop, shop): shop for shop in SHOP_DICT.keys()}


def get_mpk_code(shop_name):
    return SHOP_TO_MPK.get(shop_name, shop_name)


def load_csv(url):
    resp = requests.get(url, auth=HTTPBasicAuth(HTTP_USERNAME, HTTP_PASSWORD))
    resp.raise_for_status()
    sep = ';' if ';' in resp.text.splitlines()[0] else ','
    df = pd.read_csv(StringIO(resp.text), sep=sep, on_bad_lines='skip')
    return df


def extract_id_and_name(url):
    """
    Ze sluga URL wyciąga (ProductName, ID).
    ID = końcowe segmenty (rozdzielone '-') zawierające cyfry.
    Reszta = nazwa produktu.

    Przykłady:
      ugg-tazz-ii-damske-pantofle-bezova-1174471-san
        -> ProductName='ugg-tazz-ii-damske-pantofle-bezova', ID='1174471-san'
      jordan-spizike-low-bg-mladeznicke-tenisky-cerna-fq3950-010
        -> ProductName='jordan-spizike-low-bg-mladeznicke-tenisky-cerna', ID='fq3950-010'
      timberland-premium-6-inch-boot-panske-casual-zluta-tb1100617131
        -> ProductName='timberland-premium-6-inch-boot-panske-casual-zluta', ID='tb1100617131'
    """
    import re
    try:
        url_str = str(url).rstrip('/')
        slug = url_str.split('/')[-1]
        if len(slug) < 3 and len(url_str.split('/')) >= 2:
            slug = url_str.split('/')[-2]

        parts = slug.split('-')

        # Zbieraj od końca segmenty zawierające cyfry — to ID
        id_parts = []
        for part in reversed(parts):
            if re.search(r'\d', part):
                id_parts.insert(0, part)
            else:
                break

        if not id_parts:
            return slug, ''

        id_str = '-'.join(id_parts)
        name_str = '-'.join(parts[:len(parts) - len(id_parts)])
        return name_str, id_str
    except:
        return '', ''


def pct_diff(a, b):
    """Różnica procentowa: (a - b) / b * 100, zwraca None gdy b == 0."""
    if b == 0:
        return None
    return (a - b) / b * 100


# --- Wybór sklepów ---
selected_mpk_codes = st.multiselect(
    "Wybierz sklepy do porównania",
    list(MPK_TO_SHOP.keys()),
    default=[]
)

selected_shops = [MPK_TO_SHOP[mpk] for mpk in selected_mpk_codes]

if len(selected_shops) < 1:
    st.info("👆 Wybierz przynajmniej jeden sklep, aby rozpocząć")
    st.stop()

# --- Wczytanie danych ---
shop_data = {}
for shop_name in selected_shops:
    mpk_code = get_mpk_code(shop_name)
    with st.spinner(f'Wczytuję dane z {mpk_code}...'):
        df = load_csv(SHOP_DICT[shop_name])
        extracted = df['URL'].apply(extract_id_and_name)
        df['ProductName'] = extracted.apply(lambda x: x[0])
        df['_ID_from_url'] = extracted.apply(lambda x: x[1])
        for col in ['ID', 'Brand', 'Quantity', 'Variants']:
            if col not in df.columns:
                df[col] = ''
        # Uzupełnij ID z URL jeśli kolumna ID jest pusta
        df['ID'] = df['ID'].astype(str).str.strip()
        mask_empty = df['ID'].isin(['', 'nan', 'None'])
        df.loc[mask_empty, 'ID'] = df.loc[mask_empty, '_ID_from_url']
        df.drop(columns=['_ID_from_url'], inplace=True)
        df['Variants'] = pd.to_numeric(df['Variants'], errors='coerce').fillna(0)
        df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce').fillna(0)
        df['MPK'] = mpk_code
        shop_data[shop_name] = df

# ============================================================
# BUDOWANIE TABELI WYNIKOWEJ
# ============================================================

if len(selected_shops) == 1:
    shop_name = selected_shops[0]
    mpk_code = get_mpk_code(shop_name)
    result_final = shop_data[shop_name][['Index', 'ID', 'Brand', 'Price', 'Variants', 'Quantity', 'MPK']].copy()
    result_final['ProductName'] = shop_data[shop_name]['ProductName']

elif len(selected_shops) == 2:
    shop1, shop2 = selected_shops[0], selected_shops[1]
    mpk1, mpk2 = get_mpk_code(shop1), get_mpk_code(shop2)
    df1, df2 = shop_data[shop1], shop_data[shop2]

    # Merge: najpierw po Index, fallback po ID
    merged = pd.merge(df1, df2, on='Index', suffixes=(f'_{mpk1}', f'_{mpk2}'))
    merge_key = 'Index'

    if merged.empty:
        st.info("Brak wspólnych produktów po Index — próbuję po ID...")
        merged = pd.merge(df1, df2, on='ID', suffixes=(f'_{mpk1}', f'_{mpk2}'))
        merge_key = 'ID'

    if merged.empty:
        st.warning("Brak wspólnych produktów (Index ani ID) między wybranymi sklepami")
        st.stop()

    # --- Różnice kwotowe ---
    merged['Price_Diff'] = merged[f'Price_{mpk1}'] - merged[f'Price_{mpk2}']
    merged['Variants_Diff'] = merged[f'Variants_{mpk1}'] - merged[f'Variants_{mpk2}']
    merged['Quantity_Diff'] = merged[f'Quantity_{mpk1}'] - merged[f'Quantity_{mpk2}']

    # --- Różnice procentowe ---
    merged['Price_Diff_Pct'] = merged.apply(
        lambda r: pct_diff(r[f'Price_{mpk1}'], r[f'Price_{mpk2}']), axis=1)
    merged['Variants_Diff_Pct'] = merged.apply(
        lambda r: pct_diff(r[f'Variants_{mpk1}'], r[f'Variants_{mpk2}']), axis=1)
    merged['Quantity_Diff_Pct'] = merged.apply(
        lambda r: pct_diff(r[f'Quantity_{mpk1}'], r[f'Quantity_{mpk2}']), axis=1)

    # Wybierz ID i Brand z pierwszego sklepu
    if merge_key == 'Index':
        id_col   = f'ID_{mpk1}'   if f'ID_{mpk1}'   in merged.columns else 'ID'
        brand_col = f'Brand_{mpk1}' if f'Brand_{mpk1}' in merged.columns else 'Brand'
        index_col = 'Index'
        id_val    = merged[id_col]
    else:
        # merge po ID — Index może mieć sufiks
        index_col_name = f'Index_{mpk1}' if f'Index_{mpk1}' in merged.columns else 'Index'
        index_col = index_col_name
        id_val    = merged['ID']
        brand_col = f'Brand_{mpk1}' if f'Brand_{mpk1}' in merged.columns else 'Brand'

    result_final = pd.DataFrame({
        'Index':               merged[index_col],
        'ID':                  id_val if merge_key == 'Index' else merged['ID'],
        'Brand':               merged[brand_col],
        f'Price_{mpk1}':       merged[f'Price_{mpk1}'],
        f'Price_{mpk2}':       merged[f'Price_{mpk2}'],
        'Price_Diff (kwota)':  merged['Price_Diff'].round(2),
        'Price_Diff (%)':      merged['Price_Diff_Pct'].round(2),
        f'Variants_{mpk1}':    merged[f'Variants_{mpk1}'],
        f'Variants_{mpk2}':    merged[f'Variants_{mpk2}'],
        'Variants_Diff (szt)': merged['Variants_Diff'],
        'Variants_Diff (%)':   merged['Variants_Diff_Pct'].round(2),
        f'Quantity_{mpk1}':    merged[f'Quantity_{mpk1}'],
        f'Quantity_{mpk2}':    merged[f'Quantity_{mpk2}'],
        'Quantity_Diff (szt)': merged['Quantity_Diff'],
        'Quantity_Diff (%)':   merged['Quantity_Diff_Pct'].round(2),
        'ProductName':         merged[f'ProductName_{mpk1}'],
    })

else:
    st.info("Wybrano więcej niż 2 sklepy - wyświetlam dane dla każdego oddzielnie")
    result_final = pd.DataFrame()
    for shop_name in selected_shops:
        mpk_code = get_mpk_code(shop_name)
        df_temp = shop_data[shop_name][['Index', 'ID', 'Brand', 'Price', 'Variants', 'Quantity', 'MPK', 'ProductName']].copy()
        result_final = pd.concat([result_final, df_temp], ignore_index=True)

# ============================================================
# FILTRY
# ============================================================

columns_to_skip = ['ProductName', 'Index']
text_columns = []
numeric_columns = []

for col in result_final.columns:
    if col in columns_to_skip:
        continue
    if pd.api.types.is_numeric_dtype(result_final[col]):
        numeric_columns.append(col)
    else:
        text_columns.append(col)

if 'column_filters' not in st.session_state:
    st.session_state['column_filters'] = {}

st.markdown("---")
st.markdown("### 🔍 Filtry danych")

active_filters_count = 0
for col_name, filter_val in st.session_state['column_filters'].items():
    if col_name in text_columns and filter_val:
        active_filters_count += 1
    elif col_name in numeric_columns and filter_val:
        col_data = result_final[col_name].dropna()
        if len(col_data) > 0:
            min_val = float(col_data.min())
            max_val = float(col_data.max())
            if filter_val != (min_val, max_val):
                active_filters_count += 1

if active_filters_count > 0:
    st.success(f"✅ Aktywnych filtrów: {active_filters_count}")
else:
    st.info("💡 **Jak filtrować:** Kliknij w ekspander z nazwą kolumny (🔽) poniżej, aby ustawić filtry")

all_columns = text_columns + numeric_columns
num_cols = 4

for i in range(0, len(all_columns), num_cols):
    cols = st.columns(num_cols)
    for idx, col_name in enumerate(all_columns[i:i+num_cols]):
        with cols[idx]:
            with st.expander(f"🔽 {col_name}"):
                if col_name in text_columns:
                    unique_vals = sorted(result_final[col_name].dropna().astype(str).unique().tolist())
                    search = st.text_input("Szukaj", key=f"search_{col_name}", placeholder="Wpisz frazę...")
                    if search:
                        unique_vals = [v for v in unique_vals if search.lower() in v.lower()]
                    selected = st.multiselect(
                        "Wybierz wartości",
                        options=unique_vals,
                        default=st.session_state['column_filters'].get(col_name, []),
                        key=f"multi_{col_name}"
                    )
                    st.session_state['column_filters'][col_name] = selected

                elif col_name in numeric_columns:
                    col_data = result_final[col_name].dropna()
                    if len(col_data) > 0:
                        min_val = float(col_data.min())
                        max_val = float(col_data.max())
                        if min_val != max_val:
                            current_range = st.session_state['column_filters'].get(col_name, (min_val, max_val))
                            safe_min = max(min_val, min(current_range[0], max_val))
                            safe_max = min(max_val, max(current_range[1], min_val))
                            range_val = st.slider(
                                "Zakres",
                                min_value=min_val,
                                max_value=max_val,
                                value=(safe_min, safe_max),
                                key=f"slider_{col_name}"
                            )
                            st.session_state['column_filters'][col_name] = range_val
                            st.caption(f"Min: {range_val[0]:.2f} | Max: {range_val[1]:.2f}")

if st.button("🔄 Resetuj wszystkie filtry", use_container_width=True):
    st.session_state['column_filters'] = {}
    # Wyczyść też klucze widgetów filtrów
    for col_name in all_columns:
        for prefix in ['search_', 'multi_', 'slider_']:
            key = f"{prefix}{col_name}"
            if key in st.session_state:
                del st.session_state[key]
    st.rerun()

# Aplikowanie filtrów
filtered_df = result_final.copy()
for col_name, filter_val in st.session_state['column_filters'].items():
    if col_name in text_columns and filter_val:
        filtered_df = filtered_df[filtered_df[col_name].astype(str).isin(filter_val)]
    elif col_name in numeric_columns and filter_val:
        filtered_df = filtered_df[
            (filtered_df[col_name] >= filter_val[0]) &
            (filtered_df[col_name] <= filter_val[1])
        ]

# ============================================================
# WYŚWIETLENIE TABELI I POBIERANIE
# ============================================================

if filtered_df is not None and not filtered_df.empty:
    st.markdown("---")
    st.subheader(f"Porównanie: {', '.join(selected_mpk_codes)}")
    st.caption(f"Wyświetlono {len(filtered_df)} z {len(result_final)} produktów")

    display_columns = [col for col in filtered_df.columns if col != 'ProductName']
    display_df = filtered_df[display_columns].copy()

    edited_df = st.data_editor(
        display_df,
        use_container_width=True,
        height=500,
        num_rows="fixed",
        key="data_editor"
    )

    edited_df['ProductName'] = filtered_df['ProductName'].values

    st.markdown("---")
    st.markdown("### 📥 Pobierz dane")
    col1, col2 = st.columns(2)

    with col1:
        csv = edited_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Pobierz CSV (przefiltrowane dane)",
            data=csv,
            file_name=f"porownanie_{'_'.join(selected_mpk_codes)}_filtered.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col2:
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            edited_df.to_excel(writer, index=False, sheet_name='Porównanie')
        buffer.seek(0)
        st.download_button(
            label="📥 Pobierz XLSX (przefiltrowane dane)",
            data=buffer,
            file_name=f"porownanie_{'_'.join(selected_mpk_codes)}_filtered.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
else:
    st.warning("Brak danych po zastosowaniu filtrów. Spróbuj zmienić kryteria filtrowania.")
