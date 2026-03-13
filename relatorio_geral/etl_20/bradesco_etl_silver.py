  # bradesco_etl_silver.py
import os
import numpy as np
from io import BytesIO
from pathlib import Path
from google.cloud import storage
import pandas as pd
import re
import unicodedata
from dotenv import load_dotenv

load_dotenv('/home/gaolgo/documents/repositories/murta_engenharia/.env')

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
client = storage.Client()
bucket = client.bucket(os.getenv("GCS_BUCKET_NAME"))

def upload_to_gcs(local_path, bucket_name, key):
    # envia o arquivo local para GCS
    bucket = client.bucket(bucket_name)
    blob   = bucket.blob(key)
    blob.upload_from_filename(local_path)
    print(f"Uploaded {local_path} → gs://{bucket_name}/{key}")

def processar_arquivos(lista_caminhos):
    # 1) carregar e concatenar raw
    df_concat = pd.concat([_ler_planilha(p) for p in lista_caminhos], ignore_index=True)
    df_concat = df_concat[df_concat["Data Envio"].notna()]
    # manter uma cópia do raw antes de qualquer transformação
    
    df_concat['Período de Cobrança Inicial'] = pd.to_datetime(df_concat['Período de Cobrança Inicial'], dayfirst=True)
    df_concat['Período de Cobrança Final']   = pd.to_datetime(df_concat['Período de Cobrança Final'], dayfirst=True)

    # 1) compute inclusive internação days
    df_concat['Qtde. Diárias Geral'] = (
        df_concat['Período de Cobrança Final'] - df_concat['Período de Cobrança Inicial']).dt.days + 1
    
    raw = df_concat.copy()

    # 2) limpeza de nomes de colunas
    df_concat.rename(
        columns=lambda c: c.replace("Qtde", "Qtde.") if "Qtde" in c else c,
        inplace=True
    )
    df_concat.rename(
        columns=lambda c: c.replace("..", ".") if ".." in c else c,
        inplace=True
    )

    m        = _criar_mapeamento()
    df_long  = _reestruturar_df(df_concat, m, _COLUNAS_INDICE + ['tipo_de_internacao'])
    df_proc  = _pos_processamento(df_long)

    idx_ext  = _COLUNAS_INDICE \
                + ['tipo_de_internacao'] \
                + ["categoria", "sub_categoria", "variavel"]

    df_of    = _pivotar_ofensores(df_proc, idx_ext)
    df_of    = _processar_mes(df_of)
    df_of.rename(columns=_clean_column, inplace=True)
    df_of    = df_of[df_of["operadora"] != "NAO INFORMADO"]

    # 4) separar despesas e diárias, mantendo a coluna “categoria”

    # Step 1: identificar linhas de “Despesas”
    is_despesa = df_of["variavel"] == "Despesas"

    # Step 2: subset apenas de “Despesas”
    df_desp = (
        df_of[is_despesa]
        .drop(columns="variavel")
        .copy()
    )

    # Step 3: subset das demais linhas
    df_rest = (
        df_of[~is_despesa]
        .drop(columns="variavel")
        .copy()
    )

    # Step 4: nas não-despesas, criar sub_categoria e sobrescrever categoria
    df_dia = df_rest.assign(
        sub_categoria=df_rest["categoria"],  # guarda a categoria original
        categoria="Diárias"                  # categoria passa a ser “Diárias”
    )

    # Step 5: unir os dois dataframes
    combined = pd.concat([df_desp, df_dia], ignore_index=True)

    # Step 7: criar nova coluna “ofensor” a partir de “categoria” (mantendo “categoria”)
    combined = combined.assign(
    ofensor=np.where(
        combined["categoria"] == "Diárias",
        combined["sub_categoria"],
        combined["categoria"]))

    # Step 8: forçar tipos finais (incluindo agora “categoria” e “ofensor”)
    dtype_map = {
        'operadora': 'int64',
        'nome_operadora': 'string',
        'codigo_do_referenciado': 'int64',
        'nome_do_referenciado': 'string',
        'tipo_de_conta': 'string',
        'cidade_do_referenciado': 'string',
        'estado_do_referenciado': 'string',
        'senha': 'string',
        'data_envio': 'string',
        'categoria': 'string',
        'sub_categoria': 'string',
        'ofensor': 'string',
        'cobrado': 'float64',
        'liberado': 'float64',
        'mes': 'string',
        'mes_resumido': 'string'
    }

    ofensores = combined.astype(dtype_map)
    ofensores = ofensores.drop(columns='categoria')

    # 5) montar despesas (diárias)
    despesas = (
        df_dia
        .rename(columns={"diarias": "numero_de_diarias", "sub_categoria": "tipo_diaria"})
        .drop(columns="categoria")
        .astype({
            'operadora': 'int64',
            'nome_operadora': 'string',
            'codigo_do_referenciado': 'int64',
            'nome_do_referenciado': 'string',
            'tipo_de_conta': 'string',
            'cidade_do_referenciado': 'string',
            'estado_do_referenciado': 'string',
            'senha': 'string',
            'data_envio': 'string',
            'tipo_diaria': 'string',
            'numero_de_diarias': 'float64',
            'cobrado': 'float64',
            'liberado': 'float64',
            'mes': 'string',
            'mes_resumido': 'string'
        })
    )

    # 6) internacoes
    # 0) parse date columns once (if not already)
    
    df_concat['tipo_de_internacao'] = df_concat['tipo_de_internacao'].fillna('NÃO INFORMADO')

    # 2) now run your existing pipeline exactly as-is:
    internacao = (
        df_concat[_COLUNAS_INDICE + _COLUNAS_INTERNACAO]
        .groupby(_COLUNAS_INDICE + ["tipo_de_internacao"], as_index=False)
        .sum()
        .pipe(_processar_mes)
        .rename(columns=_clean_column)
    ).astype({
        'operadora': 'int64',
        'nome_operadora': 'string',
        'codigo_do_referenciado': 'int64',
        'nome_do_referenciado': 'string',
        'tipo_de_conta': 'string',
        'cidade_do_referenciado': 'string',
        'estado_do_referenciado': 'string',
        'senha': 'string',
        'data_envio': 'string',
        'tipo_de_internacao': 'string',
        'dias_de_internacoes_liberadas': 'int64',
        'dias_de_internacoes': 'int64',       # now comes from our new calculation
        'cobrado': 'float64',
        'liberado': 'float64',
        'mes': 'string',
        'mes_resumido': 'string'
    })


    # 7) montar tabela de pacientes
    pacientes_new = (
        raw[[
            'Nome do Referenciado', 'Estado do Referenciado',
            'Código do Referenciado', 'Nome do Segurado',
            'Senha', 'Data da Internação Solicitada',
            'Data da Internação Real'
        ]]
        .drop_duplicates()
    )

    pacientes_new_merged = (
        pacientes_new
        .merge(
            internacao[[
                'senha', 'dias_de_internacoes', 'dias_de_internacoes_liberadas'
            ]],
            left_on='Senha',
            right_on='senha',
            how='left'
        )
        .drop(columns='senha')
    )

    pacientes_new_merged['flag_mais_de_15_dias'] = np.where(
        pacientes_new_merged['dias_de_internacoes'] > 15,
        'SIM',
        'NÃO'
    )

    # 7.4) padronizar nomes das colunas em snake_case
    pacientes_new_merged.columns = [
        'nome_do_referenciado',
        'estado_do_referenciado',
        'codigo_do_referenciado',
        'nome_do_segurado',
        'senha',
        'data_da_internacao_solicitada',
        'data_da_internacao_real',
        'dias_de_internacoes',
        'dias_de_internacoes_liberadas',
        'flag_mais_de_15_dias'
    ]

    pacientes_new_merged = pacientes_new_merged.astype({
    'nome_do_referenciado': 'string',
    'estado_do_referenciado': 'string',
    'codigo_do_referenciado': 'Int64',
    'nome_do_segurado': 'string',
    'senha': 'string',
    'data_da_internacao_solicitada':'string',
    'data_da_internacao_real':'string',
    'dias_de_internacoes': 'Int64',
    'dias_de_internacoes_liberadas': 'Int64',
    'flag_mais_de_15_dias': 'category'})

    # 8) retornar todos os dataframes
    return ofensores, despesas, internacao, pacientes_new_merged

def upload_df_as_parquet_to_gcs(df: pd.DataFrame, arquivo: str, sufixo: str,
                                bucket_name='bucket_name', prefix_path: str = ""):
    
    bucket = storage.Client().bucket(bucket_name)

    base = os.path.splitext(os.path.basename(arquivo))[0]
    blob_name = f"{prefix_path.rstrip('/')}/{base}_{sufixo}.parquet".lstrip('/')

    buf = BytesIO()
    df.to_parquet(buf, index=False, compression="gzip")
    buf.seek(0)

    blob = bucket.blob(blob_name)
    blob.upload_from_file(buf, content_type="application/octet-stream")
    print(f"Uploaded gs://{bucket.name}/{blob_name}")


_CATEGORIAS_DIARIAS = [
    "Quarto/Apto", "Day Clinic", "UTI", "UI/SEMI",
    "Enfermaria", "Berçário", "Acompanhante", "Isolamento",
]
_SUBCATEGORIAS = [
    "Terapias", "Taxas/Alugueis", "Material de Consumo", "Medicamentos",
    "Gases Medicinais", "Material Especial", "Exames", "Hemoderivados", "Honorários",
]
_AREAS = ["Quarto/Enferm", "UTI/UI", "Centro Cirúrgico"]
_COLUNAS_INDICE = [
    "Operadora", "Nome Operadora", "Código do Referenciado",
    "Nome do Referenciado", "Tipo de conta",
    "Cidade do Referenciado", "Estado do Referenciado", "Senha", "Data Envio",
]
_COLUNAS_INTERNACAO = [
    "tipo_de_internacao", "dias_de_internacoes_liberadas",
    "dias_de_internacoes", "cobrado", "liberado",
]


def _encontrar_linha_cabecalho(caminho, obrig, max_linhas=10):
    xls = pd.ExcelFile(caminho, engine="openpyxl")
    prev = pd.read_excel(xls, header=None, nrows=max_linhas)
    for i in range(max_linhas):
        vals = prev.iloc[i].astype(str).tolist()
        if all(c in vals for c in obrig):
            return i
    return 0


def _ler_planilha(caminho):
    obrig = [
        "CNPJ da Empresa Auditora",
        "Nome da Empresa Auditora",
        "Operadora",
        "Nome Operadora",
        "Estipulante",
    ]
    head = _encontrar_linha_cabecalho(caminho, obrig)
    return (
        pd.read_excel(caminho, header=head, decimal=",", engine="openpyxl")
        .rename(
            columns={
                "Total de Dias de Internação Liberados": "dias_de_internacoes_liberadas",
                "Dias de Internação": "dias_de_internacoes",
                "Tipo de Internação": "tipo_de_internacao",
                "Valor Total Cobrado": "cobrado",
                "Valor Total Liberado": "liberado",
            }
        )
    )

def _criar_mapeamento():
    m = {}
    m["Qtde. Diárias Geral"] = {
        "full_cat": "Geral",  "variavel": "Diárias", "tipo": "Diarias"}
    for cat in _CATEGORIAS_DIARIAS:
        
        m[f"Qtde. Diárias {cat}"] = {"full_cat": cat, "variavel": "Diárias", "tipo": "Diarias"}
        m[f"Valor Diárias {cat} Cobrado"] = {"full_cat": cat, "variavel": "Diárias", "tipo": "Cobrado"}
        m[f"Valor Diárias {cat} Liberado"] = {"full_cat": cat, "variavel": "Diárias", "tipo": "Liberado"}
    for chave in ["Pacote", "Remoção"]:
        m[f"Qtde. {chave}"] = {"full_cat": chave, "variavel": chave, "tipo": "Diarias"}
        m[f"Valor {chave} Cobrado"] = {"full_cat": chave, "variavel": chave, "tipo": "Cobrado"}
        m[f"Valor {chave} Liberado"] = {"full_cat": chave, "variavel": chave, "tipo": "Liberado"}
    for area in _AREAS:
        for sub in _SUBCATEGORIAS:
            comp = f"{area} – {sub}"
            m[f"Qtde. Despesas {comp}"] = {"full_cat": comp, "variavel": "Despesas", "tipo": "Diarias"}
            m[f"Valor Despesas {comp} Cobrado"] = {"full_cat": comp, "variavel": "Despesas", "tipo": "Cobrado"}
            m[f"Valor Despesas {comp} Liberado"] = {"full_cat": comp, "variavel": "Despesas", "tipo": "Liberado"}
    return m

def _filtrar_colunas_permitidas(df, m, idx):
    permit = set(m) | set(idx)
    return df[[c for c in df.columns if c in permit]]

def _reestruturar_df(df, m, idx):
    df2 = _filtrar_colunas_permitidas(df, m, idx)
    recs = []
    for _, row in df2.iterrows():
        base = {c: row[c] for c in idx if c in df2}
        for col, info in m.items():
            if col in df2:
                comp = info["full_cat"]
                if " – " in comp:
                    cat, sub = comp.split(" – ", 1)
                else:
                    cat, sub = comp, comp
                recs.append(
                    {
                        **base,
                        "categoria": cat,
                        "sub_categoria": sub,
                        "variavel": info["variavel"],
                        "tipo_de_valor": info["tipo"],
                        "valor": row[col],
                    }
                )
    cols = [c for c in idx if c in df2] + ["categoria", "sub_categoria", "variavel", "tipo_de_valor", "valor"]
    return pd.DataFrame(recs, columns=cols).sort_values(cols).reset_index(drop=True)


def _pos_processamento(df):
    df = df.fillna("NAO INFORMADO")
    df["valor"] = (
        df["valor"].astype(str)
        .str.replace(",", ".", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
        .fillna(0)
    )
    return df

def _pivotar_ofensores(df, idx_ext):
    return (
        df.pivot_table(index=idx_ext, columns="tipo_de_valor", values="valor", aggfunc="sum")
        .reset_index()
        .rename_axis(None, axis=1)
    )

def _processar_mes(df):
    df = df.copy()
    df["Data Envio"] = pd.to_datetime(df["Data Envio"], format="%Y-%m-%d", errors="coerce")
    mmap = {
        1: "Janeiro", 2: "Fevereiro", 3: "Março",
        4: "Abril", 5: "Maio", 6: "Junho",
        7: "Julho", 8: "Agosto", 9: "Setembro",
        10: "Outubro", 11: "Novembro", 12: "Dezembro",
    }
    df["Mês"] = df["Data Envio"].dt.month.map(mmap)
    df["Mês Resumido"] = df["Data Envio"].dt.strftime("01/%m/%Y")
    return df

def _clean_column(col):
    na = unicodedata.normalize("NFKD", col).encode("ASCII", errors="ignore").decode("ASCII")
    clean = re.sub(r"[^0-9A-Za-z]+", "_", na)
    return clean.strip("_").lower()

from google.cloud import bigquery

def query_df_from_bigquery(sql: str, project: str = 'power-bi-data-455019') -> pd.DataFrame:
    """
    Execute a SQL query on BigQuery and return the results as a DataFrame.

    Args:
        sql: The SQL query to run.
        project: GCP project ID (uses default if None).

    Returns:
        pandas.DataFrame with the query results.
    """
    client = bigquery.Client(project=project)
    return client.query(sql).to_dataframe()
