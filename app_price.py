import re
import streamlit as st
import pandas as pd
import requests
from io import BytesIO, StringIO
from datetime import date
from requests.auth import HTTPBasicAuth

st.set_page_config(page_title="Price Checker", layout="wide")

# ============================================================
# LOGOWANIE
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

st.title("Price Checker – porównanie sklepów")

HTTP_USERNAME = st.secrets["http_auth"]["username"]
HTTP_PASSWORD = st.secrets["http_auth"]["password"]

SHOP_TO_MPK = {
    '50stylepl': 'S501', 'butysportowe': 'S503', 'sizeerpl': 'S500',
    'sizeerde': 'G500', 'sizeercz': 'CZ50', 'sizeersk': 'SK50',
    'sizeerlt': 'LT50', 'sizeerro': 'RO50', 'timberland': 'S502',
    'jdsportspl': 'S512', 'jdsportsro': 'RO55', 'jdsportssk': 'SK52',
    'jdsportshu': 'HU52', 'jdsportslt': 'LT52', 'jdsportsbg': 'BG52',
    'jdsportscz': 'CZ55', 'jdsportsua': 'UA52', 'jdsportshr': 'HR52',
    'jdsportssi': 'SI52', 'jdsportsee': 'EE52',
}

SHOP_DICT   = {name: url for name, url in st.secrets["shop_urls"].items()}
MPK_TO_SHOP = {SHOP_TO_MPK.get(s, s): s for s in SHOP_DICT.keys()}


# ────────────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────────────

def get_mpk_code(shop_name):
    return SHOP_TO_MPK.get(shop_name, shop_name)


def load_csv(url):
    resp = requests.get(url, auth=HTTPBasicAuth(HTTP_USERNAME, HTTP_PASSWORD))
    resp.raise_for_status()
    raw = resp.content
    for enc in ('utf-8-sig', 'utf-8', 'cp1250', 'iso-8859-2', 'latin-1'):
        try:
            text = raw.decode(enc)
            sep  = ';' if ';' in text.splitlines()[0] else ','
            df   = pd.read_csv(StringIO(text), sep=sep, on_bad_lines='skip')
            return df
        except (UnicodeDecodeError, Exception):
            continue
    raise ValueError("Nie można odczytać pliku CSV — nieznany encoding")


def extract_id_from_url(url):
    """
    Wyciąga ID z końca sluga URL gdy kolumna ID jest pusta.
    Przykłady:
      .../converse-chuck-taylor-m7650c   -> M7650C
      .../jordan-spizike-low-fq3950-010  -> FQ3950-010
    """
    try:
        slug  = str(url).rstrip('/').split('/')[-1]
        if len(slug) < 3:
            slug = str(url).rstrip('/').split('/')[-2]
        parts = slug.split('-')
        last_digit_idx = -1
        for i in range(len(parts) - 1, -1, -1):
            if re.search(r'\d', parts[i]):
                last_digit_idx = i
                break
        if last_digit_idx == -1:
            return ''
        return '-'.join(parts[last_digit_idx:]).upper()
    except:
        return ''


def count_sizes(val):
    try:
        if pd.isna(val) or str(val).strip() == '':
            return 0
        return len(str(val).split('|'))
    except:
        return 0


def pct_diff(a, b):
    if b == 0:
        return None
    return round((a - b) / b * 100, 2)


def color_diff(val):
    try:
        v = float(val)
    except (TypeError, ValueError):
        return ''
    if pd.isna(v) or v == 0:
        return 'color: gray'
    return 'color: red' if v > 0 else 'color: green'


# ────────────────────────────────────────────────────────────
# WYBÓR SKLEPÓW I WCZYTANIE DANYCH
# ────────────────────────────────────────────────────────────

selected_mpk_codes = st.multiselect(
    "Wybierz sklepy do porównania",
    list(MPK_TO_SHOP.keys()),
    default=[]
)
selected_shops = [MPK_TO_SHOP[m] for m in selected_mpk_codes]

if not selected_shops:
    st.info("👆 Wybierz przynajmniej jeden sklep, aby rozpocząć")
    st.stop()

shop_data = {}
for shop_name in selected_shops:
    mpk_code = get_mpk_code(shop_name)
    with st.spinner(f'Wczytuję dane z {mpk_code}...'):
        df = load_csv(SHOP_DICT[shop_name])

        for col in ['ID', 'Brand', 'Quantity', 'Variants', 'Sizes', 'CategoryName', 'Seasonality']:
            if col not in df.columns:
                df[col] = ''

        # ID: użyj z CSV, gdzie puste — wyciągnij z URL
        df['ID'] = df['ID'].astype(str).str.strip().str.upper()
        mask_empty = df['ID'].isin(['', 'NAN', 'NONE'])
        df.loc[mask_empty, 'ID'] = df.loc[mask_empty, 'URL'].apply(extract_id_from_url)

        df['Variants']   = pd.to_numeric(df['Variants'],  errors='coerce').fillna(0)
        df['Quantity']   = pd.to_numeric(df['Quantity'],   errors='coerce').fillna(0)
        df['SizesCount'] = df['Sizes'].apply(count_sizes)
        df['MPK']        = mpk_code
        shop_data[shop_name] = df

# ────────────────────────────────────────────────────────────
# BUDOWANIE TABELI WYNIKOWEJ
# ────────────────────────────────────────────────────────────

mpk1 = mpk2 = None

INFO_COLS = ['Index', 'ID', 'Brand', 'CategoryName', 'Seasonality']

if len(selected_shops) == 1:
    sn = selected_shops[0]
    df = shop_data[sn]
    result_final = df[INFO_COLS + ['Price', 'SizesCount', 'Variants', 'Quantity', 'MPK']].copy()

elif len(selected_shops) == 2:
    shop1, shop2 = selected_shops
    mpk1, mpk2   = get_mpk_code(shop1), get_mpk_code(shop2)
    st.session_state[f'len_{mpk1}'] = len(shop_data[shop1])
    st.session_state[f'len_{mpk2}'] = len(shop_data[shop2])
    df1, df2 = shop_data[shop1], shop_data[shop2]

    # Merge po Index
    merged = pd.merge(df1, df2, on='Index', suffixes=(f'_{mpk1}', f'_{mpk2}'), how='inner')

    if merged.empty:
        # Fallback: merge po ID
        st.info("Brak wspólnych po Index — próbuję po ID...")
        merged = pd.merge(
            df1.rename(columns={'Index': f'Index_{mpk1}'}),
            df2.rename(columns={'Index': f'Index_{mpk2}'}),
            on='ID', suffixes=(f'_{mpk1}', f'_{mpk2}'), how='inner'
        )
        if merged.empty:
            st.warning("Brak wspólnych produktów po Index ani ID między wybranymi sklepami.")
            st.stop()
        merged['Index'] = merged[f'Index_{mpk1}'].combine_first(merged[f'Index_{mpk2}'])

    for metric in ['Price', 'Variants', 'Quantity', 'SizesCount']:
        merged[f'{metric}_Diff']     = merged[f'{metric}_{mpk1}'] - merged[f'{metric}_{mpk2}']
        merged[f'{metric}_Diff_Pct'] = merged.apply(
            lambda r, m=metric: pct_diff(r[f'{m}_{mpk1}'], r[f'{m}_{mpk2}']), axis=1)

    def pick(col):
        c1 = f'{col}_{mpk1}'
        if c1 in merged.columns:
            return merged[c1]
        elif col in merged.columns:
            return merged[col]
        return ''

    # ID: z mpk1, uzupełnij z mpk2 gdzie puste
    if f'ID_{mpk1}' in merged.columns and f'ID_{mpk2}' in merged.columns:
        id_val = merged[f'ID_{mpk1}'].replace('', pd.NA).combine_first(merged[f'ID_{mpk2}'])
    elif 'ID' in merged.columns:
        id_val = merged['ID']
    else:
        id_val = pick('ID')

    result_final = pd.DataFrame({
        'Index':              merged['Index'],
        'ID':                 id_val,
        'Brand':              pick('Brand'),
        'CategoryName':       pick('CategoryName'),
        'Seasonality':        pick('Seasonality'),
        f'Price_{mpk1}':      merged[f'Price_{mpk1}'],
        f'Price_{mpk2}':      merged[f'Price_{mpk2}'],
        'Price_Diff':         merged['Price_Diff'].round(2),
        'Price_Diff_%':       merged['Price_Diff_Pct'],
        f'SizesCount_{mpk1}': merged[f'SizesCount_{mpk1}'],
        f'SizesCount_{mpk2}': merged[f'SizesCount_{mpk2}'],
        'SizesCount_Diff':    merged['SizesCount_Diff'],
        'SizesCount_Diff_%':  merged['SizesCount_Diff_Pct'],
        f'Variants_{mpk1}':   merged[f'Variants_{mpk1}'],
        f'Variants_{mpk2}':   merged[f'Variants_{mpk2}'],
        'Variants_Diff':      merged['Variants_Diff'],
        'Variants_Diff_%':    merged['Variants_Diff_Pct'],
        f'Quantity_{mpk1}':   merged[f'Quantity_{mpk1}'],
        f'Quantity_{mpk2}':   merged[f'Quantity_{mpk2}'],
        'Quantity_Diff':      merged['Quantity_Diff'],
        'Quantity_Diff_%':    merged['Quantity_Diff_Pct'],
    })

else:
    st.info("Wybrano więcej niż 2 sklepy — wyświetlam dane dla każdego oddzielnie")
    result_final = pd.concat(
        [shop_data[s][INFO_COLS + ['Price', 'SizesCount', 'Variants', 'Quantity', 'MPK']]
         for s in selected_shops],
        ignore_index=True
    )

# ────────────────────────────────────────────────────────────
# FILTRY
# ────────────────────────────────────────────────────────────
skip_filter  = ['Index']
text_cols    = [c for c in result_final.columns
                if c not in skip_filter and not pd.api.types.is_numeric_dtype(result_final[c])]
numeric_cols = [c for c in result_final.columns
                if c not in skip_filter and pd.api.types.is_numeric_dtype(result_final[c])]
all_columns  = text_cols + numeric_cols

if 'column_filters' not in st.session_state:
    st.session_state['column_filters'] = {}

st.markdown("---")
st.markdown("### 🔍 Filtry danych")

active = 0
for cn, fv in st.session_state['column_filters'].items():
    if cn in text_cols and fv:
        active += 1
    elif cn in numeric_cols and fv:
        cd = result_final[cn].dropna()
        if len(cd) and fv != (float(cd.min()), float(cd.max())):
            active += 1

if active:
    st.success(f"✅ Aktywnych filtrów: {active}")
else:
    st.info("💡 Kliknij ekspander kolumny poniżej, aby filtrować")

for i in range(0, len(all_columns), 4):
    cols = st.columns(4)
    for idx, cn in enumerate(all_columns[i:i+4]):
        with cols[idx]:
            with st.expander(f"🔽 {cn}"):
                if cn in text_cols:
                    uv = sorted(result_final[cn].dropna().astype(str).unique())
                    srch = st.text_input("Szukaj", key=f"search_{cn}", placeholder="Wpisz frazę...")
                    if srch:
                        uv = [v for v in uv if srch.lower() in v.lower()]
                    sel = st.multiselect(
                        "Wartości", options=uv,
                        default=st.session_state['column_filters'].get(cn, []),
                        key=f"multi_{cn}"
                    )
                    st.session_state['column_filters'][cn] = sel
                else:
                    cd = result_final[cn].dropna()
                    if len(cd):
                        mn, mx = float(cd.min()), float(cd.max())
                        if mn != mx:
                            cur  = st.session_state['column_filters'].get(cn, (mn, mx))
                            s_mn = max(mn, min(cur[0], mx))
                            s_mx = min(mx, max(cur[1], mn))
                            rv   = st.slider("Zakres", min_value=mn, max_value=mx,
                                             value=(s_mn, s_mx), key=f"slider_{cn}")
                            st.session_state['column_filters'][cn] = rv
                            st.caption(f"{rv[0]:.2f} … {rv[1]:.2f}")

if st.button("🔄 Resetuj wszystkie filtry", use_container_width=True):
    keys_to_delete = [k for k in st.session_state.keys()
                      if k.startswith(('search_', 'multi_', 'slider_'))]
    for k in keys_to_delete:
        del st.session_state[k]
    st.session_state['column_filters'] = {}
    st.rerun()

# aplikuj filtry
filtered_df = result_final.copy()
for cn, fv in st.session_state['column_filters'].items():
    if cn in text_cols and fv:
        filtered_df = filtered_df[filtered_df[cn].astype(str).isin(fv)]
    elif cn in numeric_cols and fv:
        filtered_df = filtered_df[(filtered_df[cn] >= fv[0]) & (filtered_df[cn] <= fv[1])]

# ────────────────────────────────────────────────────────────
# TABELA Z KOLOROWANIEM
# ────────────────────────────────────────────────────────────
if filtered_df is not None and not filtered_df.empty:
    st.markdown("---")
    st.subheader(f"Porównanie: {', '.join(selected_mpk_codes)}")
    st.caption(f"Wyświetlono {len(filtered_df)} z {len(result_final)} produktów")

    diff_cols = [c for c in filtered_df.columns if 'Diff' in c]

    format_rules = {}
    for col in filtered_df.columns:
        if pd.api.types.is_numeric_dtype(filtered_df[col]):
            if 'Price' in col and 'Diff' not in col:
                format_rules[col] = "{:.2f}"
            elif 'Price' in col and 'Diff' in col and '%' not in col:
                format_rules[col] = "{:+.2f}"
            elif col.endswith('%'):
                format_rules[col] = "{:+.1f}%"
            elif 'Diff' in col:
                format_rules[col] = "{:+.0f}"

    try:
        styled = filtered_df.style.map(color_diff, subset=diff_cols)
    except AttributeError:
        styled = filtered_df.style.applymap(color_diff, subset=diff_cols)
    styled = styled.format(format_rules, na_rep='—')

    pinned_config = {
        "Index":        st.column_config.Column(pinned=True),
        "ID":           st.column_config.Column(pinned=True),
        "Brand":        st.column_config.Column(pinned=True),
        "CategoryName": st.column_config.Column(pinned=True),
        "Seasonality":  st.column_config.Column(pinned=True),
    }

    st.dataframe(
        styled,
        use_container_width=True,
        height=520,
        column_config=pinned_config
    )

    # ────────────────────────────────────────────────────────
    # PODSUMOWANIE (tylko 2 sklepy)
    # ────────────────────────────────────────────────────────
    if len(selected_shops) == 2 and mpk1 and mpk2 and 'Price_Diff' in filtered_df.columns:
        st.markdown("---")
        st.markdown("### 📊 Podsumowanie wspólnych produktów")

        total     = len(filtered_df)
        orig_len1 = st.session_state.get(f'len_{mpk1}', total)
        orig_len2 = st.session_state.get(f'len_{mpk2}', total)
        pct_of_1  = round(total / orig_len1 * 100, 2) if orig_len1 else 0
        pct_of_2  = round(total / orig_len2 * 100, 2) if orig_len2 else 0

        cheaper_mask = filtered_df['Price_Diff'] < 0
        dearer_mask  = filtered_df['Price_Diff'] > 0
        equal_mask   = filtered_df['Price_Diff'] == 0

        cheaper = int(cheaper_mask.sum())
        dearer  = int(dearer_mask.sum())
        equal   = int(equal_mask.sum())

        pct_c = round(cheaper / total * 100, 1) if total else 0
        pct_d = round(dearer  / total * 100, 1) if total else 0
        pct_e = round(equal   / total * 100, 1) if total else 0

        avg_diff_c = filtered_df.loc[cheaper_mask, 'Price_Diff'].mean() if cheaper else 0
        avg_diff_d = filtered_df.loc[dearer_mask,  'Price_Diff'].mean() if dearer  else 0

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Wspólnych produktów", total)
            st.caption(f"({pct_of_1}% {mpk1}, {pct_of_2}% {mpk2})")
        with c2:
            st.metric(f"🟢 {mpk1} tańszy niż {mpk2}", cheaper)
            st.caption(f"{pct_c}% produktów\n\nŚr. taniej o: {avg_diff_c:.2f}")
        with c3:
            st.metric(f"🔴 {mpk1} droższy niż {mpk2}", dearer)
            st.caption(f"{pct_d}% produktów\n\nŚr. drożej o: +{avg_diff_d:.2f}")
        with c4:
            st.metric("⚪ Równe ceny", equal)
            st.caption(f"{pct_e}% produktów")

    # ────────────────────────────────────────────────────────
    # POBIERANIE – tylko XLSX z datą
    # ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📥 Pobierz dane")

    today_str = date.today().strftime('%Y-%m-%d')
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        filtered_df.to_excel(writer, index=False, sheet_name='Porównanie')
    buffer.seek(0)

    st.download_button(
        label="📥 Pobierz XLSX",
        data=buffer,
        file_name=f"porownanie_{'_'.join(selected_mpk_codes)}_{today_str}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )
else:
    st.warning("Brak danych po zastosowaniu filtrów. Spróbuj zmienić kryteria filtrowania.")
