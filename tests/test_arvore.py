import math

from dengue.ml.features import FEATURES, alvo_sinan, features_paciente, idade_em_anos, preparar_sinan
from dengue.ml.predict import PreditorArvore
from dengue.ml.sintetico import gerar
from dengue.ml.train_tree import treinar
from tests.conftest import dados_com
import pandas as pd


def test_idade_sinan_decodifica_unidades():
    s = idade_em_anos(pd.Series([4034, 3006, 2010, 1012, 9999, None]))
    assert s.iloc[0] == 34 and s.iloc[1] == 0.5 and round(s.iloc[2], 3) == round(10 / 365.25, 3)
    assert s.iloc[3] < 0.01 and math.isnan(s.iloc[4]) and math.isnan(s.iloc[5])


def test_preparo_sem_vazamento_e_ignorado_vira_nan():
    df = gerar(1000)
    X = preparar_sinan(df)
    assert list(X.columns) == FEATURES
    assert not any(c.startswith(("ALRM", "GRAV")) or c in ("CLASSI_FIN", "HOSPITALIZ", "EVOLUCAO") for c in X.columns)
    assert X["vomito"].isna().any()            # código 9 -> NaN (desconhecido != não)
    elegiveis, y = alvo_sinan(df)
    assert set(df.loc[elegiveis, "CLASSI_FIN"].unique()) <= {10, 11, 12}


def test_treino_e_predicao_ponta_a_ponta(tmp_path):
    saida = tmp_path / "m.joblib"
    pacote = treinar(gerar(6000), saida, auc_minimo=0.6, sintetico=False)   # base sintética tem sinal
    assert pacote["metricas"]["auc_teste"] > 0.6 and saida.exists()
    prev = PreditorArvore.carregar(saida)
    assert prev.utilizavel
    alto = dados_com(identificacao__idade=80, historico__diabetes="sim", sintomas__vomitos="sim", evolucao__dias_sintomas=6)
    baixo = dados_com(identificacao__idade=25, evolucao__dias_sintomas=1)
    assert prev.prever(alto).probabilidade > prev.prever(baixo).probabilidade
    # campos desconhecidos entram como NaN (sem erro)
    from dengue.schemas import DadosPaciente
    p = prev.prever(DadosPaciente())
    assert p.n_variaveis_ausentes == len(FEATURES) and 0 <= p.probabilidade <= 1


def test_modelo_sintetico_ou_inviavel_nao_e_utilizavel(tmp_path):
    treinar(gerar(3000), tmp_path / "s.joblib", auc_minimo=0.6, sintetico=True)
    assert not PreditorArvore.carregar(tmp_path / "s.joblib").utilizavel
    treinar(gerar(3000), tmp_path / "i.joblib", auc_minimo=0.99, sintetico=False)
    p = PreditorArvore.carregar(tmp_path / "i.joblib")
    assert not p.utilizavel and "insuficiente" in p.motivo_nao_utilizavel


def test_features_paciente_mapeia_tudo():
    f = features_paciente(dados_com(sintomas__dor_cabeca="sim"))
    assert set(f) == set(FEATURES) and f["cefaleia"] == 1.0 and f["mialgia"] == 0.0
