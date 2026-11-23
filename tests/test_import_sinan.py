import csv
import zipfile

import pytest

from dengue.ml.features import COLUNAS_SINAN
from dengue.ml.import_sinan import COLUNAS, linhas_csv, sha256_arquivo
from dengue.ml.import_sinan_rest import selecionar
from dengue.ml.train_tree import carregar_amostra_aleatoria


def test_importacao_aceita_ausencia_de_numero_notificacao(tmp_path):
    csv_path = tmp_path / "amostra.csv"
    campos = [c for c in COLUNAS_SINAN if c != "NU_NOTIFIC"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerow({"CLASSI_FIN": "10", "CS_SEXO": "F"})
    zip_path = tmp_path / "amostra.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.write(csv_path, arcname="amostra.csv")
    linhas = list(linhas_csv(zip_path))
    assert len(linhas) == 1
    assert linhas[0][1][COLUNAS.index("NU_NOTIFIC")] is None
    assert linhas[0][1][COLUNAS.index("CLASSI_FIN")] == "10"
    assert len(sha256_arquivo(zip_path)) == 64


def test_importacao_recusa_coluna_obrigatoria_ausente(tmp_path):
    p = tmp_path / "incompleto.csv"
    p.write_text("CLASSI_FIN\n10\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Colunas ausentes"):
        list(linhas_csv(p))


def test_amostra_aleatoria_reproduzivel_e_cobre_arquivo(tmp_path):
    p = tmp_path / "dados.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["CLASSI_FIN", "NU_IDADE_N"])
        w.writeheader()
        for i in range(1000):
            w.writerow({"CLASSI_FIN": 10, "NU_IDADE_N": 4000 + i})
    a = carregar_amostra_aleatoria(str(p), 50)
    b = carregar_amostra_aleatoria(str(p), 50)
    assert a.equals(b) and len(a) == 50
    assert a["NU_IDADE_N"].max() > 4500


def test_selecao_rest_reproduzivel(tmp_path):
    p = tmp_path / "amostra.csv"
    campos = [c for c in COLUNAS_SINAN if c != "NU_NOTIFIC"]
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        for _ in range(100):
            w.writerow({"CLASSI_FIN": "10"})
    a = list(selecionar(p, 0.1))
    assert a == list(selecionar(p, 0.1))
    assert 0 < len(a) < 100
