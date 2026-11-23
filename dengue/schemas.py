"""Modelos de dados (Pydantic) compartilhados por todos os componentes.

Ponto central: toda informação clínica é tri-estado (Resposta):
  SIM / NAO / NAO_INFORMADO  ->  "não informar" NUNCA significa "não tem".
"""
from __future__ import annotations

import unicodedata
import uuid
from datetime import date, datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, ValidationError


class Resposta(str, Enum):
    SIM = "sim"
    NAO = "nao"
    NAO_INFORMADO = "nao_informado"


NI = Resposta.NAO_INFORMADO


# --------------------------------------------------------------------------- #
# Dados coletados pelo Agente 1
# --------------------------------------------------------------------------- #
class Identificacao(BaseModel):
    idade: Optional[int] = Field(None, ge=0, le=120)
    sexo: Optional[Literal["F", "M"]] = None
    peso_kg: Optional[float] = Field(None, gt=0, le=400)
    gravidez: Resposta = NI


class Evolucao(BaseModel):
    dias_sintomas: Optional[int] = Field(None, ge=0, le=60)
    data_inicio: Optional[date] = None
    febre: Resposta = NI
    febre_max_c: Optional[float] = Field(None, ge=34, le=43)
    febre_continua: Resposta = NI  # SIM = ainda está com febre; NAO = já passou
    tendencia: Optional[Literal["melhorando", "piorando", "igual"]] = None


class Sintomas(BaseModel):
    dor_cabeca: Resposta = NI
    dor_atras_olhos: Resposta = NI
    dor_muscular: Resposta = NI
    dor_articulacoes: Resposta = NI
    nausea: Resposta = NI
    vomitos: Resposta = NI
    manchas_pele: Resposta = NI
    cansaco_prostracao: Resposta = NI
    falta_apetite: Resposta = NI


class SinaisAlerta(BaseModel):
    dor_abdominal_intensa: Resposta = NI
    vomitos_persistentes: Resposta = NI
    sangramento_mucosa: Resposta = NI
    sangue_fezes_outro_sangramento: Resposta = NI
    tontura_desmaio: Resposta = NI
    sonolencia_irritabilidade: Resposta = NI
    dificuldade_respirar: Resposta = NI
    inchaco_abdominal: Resposta = NI


class Historico(BaseModel):
    doenca_preexistente: Resposta = NI
    doenca_preexistente_detalhe: Optional[str] = None
    uso_continuo_medicamento: Resposta = NI
    medicamento_detalhe: Optional[str] = None
    alergia: Resposta = NI
    alergia_detalhe: Optional[str] = None
    teve_dengue_antes: Resposta = NI
    hipertensao: Resposta = NI
    diabetes: Resposta = NI
    doenca_cardiaca: Resposta = NI
    doenca_renal: Resposta = NI
    doenca_pulmonar: Resposta = NI
    doenca_hepatica: Resposta = NI
    doenca_hematologica: Resposta = NI
    doenca_autoimune: Resposta = NI
    anticoagulantes: Resposta = NI
    outra_condicao: Resposta = NI
    outra_condicao_detalhe: Optional[str] = None


class Exames(BaseModel):
    hemograma_disponivel: Resposta = NI
    hematocrito_pct: Optional[float] = Field(None, ge=5, le=80)
    plaquetas: Optional[int] = Field(None, ge=0, le=2_000_000)


GRUPOS_DADOS = ("identificacao", "evolucao", "sintomas", "sinais_alerta", "historico", "exames")


def _limpar(valor):
    if isinstance(valor, str):
        s = unicodedata.normalize("NFKD", valor).encode("ascii", "ignore").decode()
        return s.strip().lower()
    return valor


class DadosPaciente(BaseModel):
    identificacao: Identificacao = Field(default_factory=Identificacao)
    evolucao: Evolucao = Field(default_factory=Evolucao)
    sintomas: Sintomas = Field(default_factory=Sintomas)
    sinais_alerta: SinaisAlerta = Field(default_factory=SinaisAlerta)
    historico: Historico = Field(default_factory=Historico)
    exames: Exames = Field(default_factory=Exames)
    # perguntado, mas sem resposta clara / "não sei"
    campos_sem_resposta: list[str] = Field(default_factory=list)

    # -- acesso por caminho "grupo.campo" ---------------------------------- #
    def obter(self, caminho: str):
        grupo, campo = caminho.split(".", 1)
        return getattr(getattr(self, grupo), campo)

    def resolvido(self, caminho: str) -> bool:
        v = self.obter(caminho)
        return v is not None and v != NI

    def caminhos(self) -> list[str]:
        return [
            f"{g}.{c}" for g in GRUPOS_DADOS for c in type(getattr(self, g)).model_fields
        ]

    def definir(self, caminho: str, valor) -> bool:
        """Atualiza um campo validando com Pydantic. Retorna True se aplicou.
        Valores vazios/NAO_INFORMADO nunca sobrescrevem uma informação existente."""
        try:
            grupo, campo = caminho.split(".", 1)
            sub = getattr(self, grupo)
            if campo not in type(sub).model_fields:
                return False
        except (ValueError, AttributeError):
            return False
        if isinstance(valor, bool):
            valor = "sim" if valor else "nao"
        if (
            valor is None
            or valor == ""
            or valor == NI
            or _limpar(valor) in ("nao_informado", "null", "none")
        ):
            return False
        for candidato in (valor, _limpar(valor)):
            try:
                novo = type(sub).model_validate({**sub.model_dump(), campo: candidato})
            except ValidationError:
                continue
            setattr(self, grupo, novo)
            return True
        return False


# --------------------------------------------------------------------------- #
# Estado da conversa (Agente 1 / Orquestrador)
# --------------------------------------------------------------------------- #
class Mensagem(BaseModel):
    role: Literal["assistant", "user"]
    content: str


class ColetaEstado(BaseModel):
    sessao_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    criada_em: datetime = Field(default_factory=datetime.now)
    dados: DadosPaciente = Field(default_factory=DadosPaciente)
    perguntadas: list[str] = Field(default_factory=list)
    tentativas: dict[str, int] = Field(default_factory=dict)
    inconsistencias: list[str] = Field(default_factory=list)
    atual: Optional[str] = None
    mensagens: list[Mensagem] = Field(default_factory=list)
    concluida: bool = False
    interrompida: bool = False


# --------------------------------------------------------------------------- #
# Resultados das etapas seguintes
# --------------------------------------------------------------------------- #
class Grupo(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"


DESCRICAO_GRUPOS = {
    "A": "Sem sinais de alarme, condições especiais, comorbidades ou risco social identificados.",
    "B": "Sem sinais de alarme, mas com condição especial, comorbidade, informação essencial faltante ou situação que exige avaliação clínica.",
    "C": "Presença de sinal(is) de alarme, sem sinais de gravidade identificados.",
    "D": "Indício de sinal de gravidade (ex.: dificuldade respiratória, sangramento importante).",
}


class ResultadoSeguranca(BaseModel):
    campos_presentes: list[str] = Field(default_factory=list)
    sinais_presentes: list[str] = Field(default_factory=list)  # rótulos legíveis
    possivel_gravidade: bool = False
    nao_respondidos: list[str] = Field(default_factory=list)
    interromper: bool = False


class ClassificacaoRegras(BaseModel):
    grupo: Grupo
    justificativas: list[str] = Field(default_factory=list)
    atencoes: list[str] = Field(default_factory=list)
    dados_incompletos: list[str] = Field(default_factory=list)


class PrevisaoArvore(BaseModel):
    alvo: str
    probabilidade: float
    prevalencia_base: float
    razao_vs_base: float
    nivel: Literal["abaixo da média", "semelhante à média", "acima da média"]
    auc_teste: Optional[float] = None
    n_variaveis_ausentes: int = 0
    avisos: list[str] = Field(default_factory=list)


class AnaliseInicial(BaseModel):
    classificacao: ClassificacaoRegras
    arvore: Optional[PrevisaoArvore] = None
    arvore_motivo_ausencia: Optional[str] = None
    resumo: str = ""
    referencias_documentais: list[str] = Field(default_factory=list)


class ItemOrientacao(BaseModel):
    texto: str
    refs: list[int] = Field(default_factory=list)


class Referencia(BaseModel):
    numero: int
    fonte: str
    pagina: Optional[int] = None
    trecho: str = ""


class Orientacoes(BaseModel):
    orientacoes: list[ItemOrientacao] = Field(default_factory=list)
    quando_procurar_atendimento: list[ItemOrientacao] = Field(default_factory=list)
    referencias: list[Referencia] = Field(default_factory=list)
    contexto_insuficiente: bool = False
    aviso: Optional[str] = None


class ResultadoSessao(BaseModel):
    sessao_id: str
    interrompido: bool
    seguranca: ResultadoSeguranca
    analise: AnaliseInicial
    orientacoes: Orientacoes
    mensagem_urgente: Optional[str] = None
    resumo_ia: Optional[str] = None
    relatorio_md: str = ""
    etapas: list[str] = Field(default_factory=list)
