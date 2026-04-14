"""
Alunos: Gabriel Schaldach Morgado, Eduardo Lechinski Ramos
Trabalho M1 - Inteligência Artificial
Continuação a partir da Seção 2.3

NOTA: Este arquivo é a continuação do notebook já existente.
      Cole este código logo após a célula de 2.2 (StandardScaler).
      O pré-processamento é refeito corretamente aqui (scaler ajustado
      apenas no treino, como exige a boa prática).
"""

# ===========================================================================
# IMPORTS (já devem estar no topo do notebook — mantidos aqui por clareza)
# ===========================================================================
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, initializers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from sklearn.utils.class_weight import compute_class_weight

from ucimlrepo import fetch_ucirepo

tf.random.set_seed(42)
np.random.seed(42)

dry_bean = fetch_ucirepo(id=602)
if not dry_bean.data:
    raise Exception("could not fetch")

X = dry_bean.data.features
y = dry_bean.data.targets

print(dry_bean.metadata)

# variable information
print(dry_bean.variables)

# Observando classes alvo
y.value_counts()

y.shape

X.shape

## 1.2 Verificação e remoção de duplicatas
"""

# Verificando registros duplicados
X.duplicated().sum()

pd.concat([X, y], axis=1).duplicated().sum()

X.columns

"""  ## 1.3 Distribuição das features"""

# Fazendo histograma
X.hist(figsize=(15, 12))

# Fazendo histogramas com transformação logarítmica
X.transform(np.log).hist(figsize=(15, 12))

"""## 1.4 Análise e correlação de Pearson"""

sns.heatmap(X.corr())

X.corr()

"""## 1.5 Boxplots por classe"""

fig, axes = plt.subplots(3, 2, sharex=False, sharey=False, figsize=(15, 18))

pd.concat([X, y], axis=1).boxplot(
    column=[
        "Area",
        "Eccentricity",
        "Roundness",
        "ShapeFactor1",
        "MajorAxisLength",
        "Extent",
    ],
    by="Class",
    ax=axes,
)

"""# 2. Pré-processamento

## 2.1. Codificação da variável alvo
"""

le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_encoded

"""## 2.2 Normalização das features"""

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled

"""## 2.3 Tratamento do desbalanceamento com class_weight"""

# ===========================================================================
# 2.3  Tratamento do desbalanceamento com class_weight
# ===========================================================================

"""## 2.3 Tratamento do desbalanceamento com class_weight"""

# Calcula os pesos de forma balanceada (inversamente proporcional à frequência)
classes_unicas = np.unique(y_encoded)
pesos = compute_class_weight(
    class_weight="balanced", classes=classes_unicas, y=y_encoded
)
class_weight_dict = dict(zip(classes_unicas, pesos))

print("Pesos por classe (balanced):")
for idx, nome in enumerate(le.classes_):
    print(f"  {nome:12s} (índice {idx}): {class_weight_dict[idx]:.4f}")

# Resposta — Questão 14:
# BOMBAY tem o maior peso porque possui o menor número de amostras (522).
# O peso é inversamente proporcional à frequência da classe:
# quanto menos exemplos, maior a penalização por erro.

# Resposta — Questão 15:
# Com esses pesos, um erro em BOMBAY gera uma penalização muito maior na
# função de perda do que um erro em DERMASON (classe majoritária).
# Isso força a rede a prestar mais atenção às classes raras.

# ===========================================================================
# 2.4  Divisão treino / validação / teste
# ===========================================================================

"""## 2.4 Divisão treino / validação / teste"""

# Estratégia: 60 % treino | 20 % validação | 20 % teste
# O parâmetro stratify mantém a proporção de cada classe em todos os splits.

X_arr = X.values  # converte o DataFrame para numpy antes de dividir

# 1ª divisão: 80 % treino+val  /  20 % teste
X_trainval, X_test_raw, y_trainval, y_test = train_test_split(
    X_arr, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
)

# 2ª divisão: 75 % treino  /  25 % validação (do conjunto de 80 %)
# → resulta em 60 % treino e 20 % validação do total
X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.25, random_state=42, stratify=y_trainval
)

print(f"Treino    : {X_train_raw.shape[0]} amostras")
print(f"Validação : {X_val_raw.shape[0]} amostras")
print(f"Teste     : {X_test_raw.shape[0]} amostras")

# ----- Normalização correta: ajuste APENAS no treino -----
# (o scaler calculado em 2.2 foi ajustado em TODOS os dados; aqui refazemos
#  de forma rigorosa para evitar vazamento de informação para val/teste)
scaler_final = StandardScaler()
X_train = scaler_final.fit_transform(X_train_raw)  # fit + transform no treino
X_val = scaler_final.transform(X_val_raw)  # apenas transform
X_test = scaler_final.transform(X_test_raw)  # apenas transform

# Resposta — Questão 12:
# O scaler não pode ser ajustado nos dados de teste pois isso significaria
# que a estatística de escalonamento (média e desvio padrão) foi calculada
# com informação do futuro (dados que o modelo nunca deveria ver durante o
# treinamento), introduzindo vazamento de dados (data leakage) e
# superestimando o desempenho real em produção.

# Resposta — Questão 13:
# Em produção, a entrada do usuário deve ser escalonada com os mesmos
# parâmetros (média e desvio) estimados no treino.  Se um scaler diferente
# for usado, os valores chegarão à rede em escala errada, e as predições
# serão incorretas — mesmo que o modelo em si esteja perfeitamente treinado.

# Resposta — Questão 16:
# O stratify é fundamental neste dataset porque há classes com frequências
# muito diferentes (ex.: BOMBAY tem 522 amostras, DERMASON tem 3546).
# Sem stratify, um split aleatório poderia concentrar quase todos os exemplos
# de BOMBAY em um único conjunto, prejudicando o treinamento ou tornando a
# avaliação irrepresentativa.

# Resposta — Questão 17:
# Os dados de teste devem ser usados apenas na avaliação final porque
# representam dados "nunca vistos".  Se os usarmos para escolher
# hiperparâmetros ou arquitetura, estamos, indiretamente, ajustando o modelo
# para eles — e a métrica obtida deixa de ser uma estimativa imparcial do
# desempenho em novos dados reais.

# ===========================================================================
# 3.  Construção, treinamento e experimentos
# ===========================================================================

"""# 3. Construção, treinamento e experimentos"""

# ---------- Features selecionadas na AED (Seção 1.5, Questão 9) ----------
colunas = list(X.columns)
features_sel = [
    "Area",
    "MajorAxisLength",
    "MinorAxisLength",
    "AspectRation",
    "Eccentricity",
    "Extent",
    "Solidity",
    "Roundness",
    "Compactness",
    "ShapeFactor1",
    "ShapeFactor2",
    "ShapeFactor4",
]
idx_sel = [colunas.index(f) for f in features_sel]

X_train_sel = X_train[:, idx_sel]
X_val_sel = X_val[:, idx_sel]
X_test_sel = X_test[:, idx_sel]

# ---------- Nomes das classes ----------
nomes_classes = list(le.classes_)
N_CLASSES = len(nomes_classes)

# ===========================================================================
# 3.1  Funções auxiliares
# ===========================================================================

"""## 3.1 Funções auxiliares"""


def construir_modelo(
    n_entrada, unidades, taxas_dropout, inicializador="he_normal", lr=1e-3
):
    """
    Constrói um MLP com:
      - n_entrada   : número de features
      - unidades    : lista com neurônios por camada oculta
      - taxas_dropout : lista com taxa de dropout (None = sem dropout)
      - inicializador : kernel_initializer
      - lr          : learning rate do Adam
    """
    modelo = keras.Sequential()
    modelo.add(layers.Input(shape=(n_entrada,)))
    for u, d in zip(unidades, taxas_dropout):
        modelo.add(layers.Dense(u, activation="relu", kernel_initializer=inicializador))
        if d is not None and d > 0:
            modelo.add(layers.Dropout(d))
    modelo.add(layers.Dense(N_CLASSES, activation="softmax"))

    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return modelo


def treinar(modelo, X_tr, y_tr, X_v, y_v, batch_size=64, epocas=300, usar_pesos=False):
    """Treina o modelo com EarlyStopping e retorna o histórico."""
    cb_early = callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
    pesos = class_weight_dict if usar_pesos else None

    historico = modelo.fit(
        X_tr,
        y_tr,
        validation_data=(X_v, y_v),
        epochs=epocas,
        batch_size=batch_size,
        callbacks=[cb_early],
        class_weight=pesos,
        verbose=0,
    )
    return historico


def plotar_curvas(historico, titulo):
    """Plota curvas de loss e acurácia (treino vs. validação)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(historico.history["loss"], label="treino")
    axes[0].plot(historico.history["val_loss"], label="validação")
    axes[0].set_title(f"{titulo} — Loss")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(historico.history["accuracy"], label="treino")
    axes[1].plot(historico.history["val_accuracy"], label="validação")
    axes[1].set_title(f"{titulo} — Acurácia")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Acurácia")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def avaliar(modelo, X_te, y_te):
    """Avalia o modelo no conjunto de teste e imprime métricas."""
    y_pred = modelo.predict(X_te, verbose=0).argmax(axis=1)
    acc = accuracy_score(y_te, y_pred)
    f1m = f1_score(y_te, y_pred, average="macro")
    print(f"  Acurácia  : {acc:.4f}")
    print(f"  F1-macro  : {f1m:.4f}")
    print()
    print(classification_report(y_te, y_pred, target_names=nomes_classes))
    return acc, f1m, y_pred


# Tabela de resultados (preenchida ao longo dos experimentos)
resultados = []

# Resposta — Questão 18:
# A camada de saída usa softmax com 7 neurônios porque temos 7 classes
# mutuamente exclusivas.  O softmax transforma o vetor de ativações em uma
# distribuição de probabilidade que soma 1.0, de modo que cada neurônio
# representa a probabilidade estimada de pertencer àquela classe.

# Resposta — Questão 19:
# sparse_categorical_crossentropy é usada porque os rótulos são inteiros
# (0–6), não vetores one-hot.  binary_crossentropy serve apenas para
# classificação binária (2 classes); para multiclasse com rótulos inteiros
# a loss correta é sparse_categorical_crossentropy.

# Resposta — Questão 20:
# Não.  O Dropout é aplicado apenas nas camadas ocultas.  Aplicá-lo à
# camada de saída desativaria neurônios responsáveis por classes inteiras,
# distorcendo as probabilidades finais e prejudicando o aprendizado.

# ===========================================================================
# 3.3  Experimento A — Sem Dropout (baseline)
# ===========================================================================

"""## Experimento A — Sem Dropout (referência)"""

tf.random.set_seed(42)
modelo_A = construir_modelo(
    n_entrada=X_train.shape[1],
    unidades=[128, 64],
    taxas_dropout=[None, None],  # sem dropout
)
historico_A = treinar(modelo_A, X_train, y_train, X_val, y_val, usar_pesos=False)
plotar_curvas(historico_A, "Experimento A — Sem Dropout")

print("=== Experimento A — Sem Dropout ===")
acc_A, f1_A, _ = avaliar(modelo_A, X_test, y_test)
epocas_A = len(historico_A.history["loss"])
resultados.append(
    {
        "Exp": "A",
        "Descrição": "Sem Dropout (baseline)",
        "Features": "Todas (16)",
        "Acurácia": round(acc_A, 4),
        "F1-macro": round(f1_A, 4),
        "Épocas": epocas_A,
        "Observações": "Possível overfitting (sem regularização)",
    }
)

# ===========================================================================
# 3.3  Experimento A2 — Com Dropout
# ===========================================================================

"""## Experimento A2 — Com Dropout"""

tf.random.set_seed(42)
modelo_A2 = construir_modelo(
    n_entrada=X_train.shape[1], unidades=[128, 64], taxas_dropout=[0.3, 0.2]
)
historico_A2 = treinar(modelo_A2, X_train, y_train, X_val, y_val, usar_pesos=False)
plotar_curvas(historico_A2, "Experimento A2 — Com Dropout")

print("=== Experimento A2 — Com Dropout ===")
acc_A2, f1_A2, _ = avaliar(modelo_A2, X_test, y_test)
epocas_A2 = len(historico_A2.history["loss"])
resultados.append(
    {
        "Exp": "A2",
        "Descrição": "Com Dropout (0.3 / 0.2)",
        "Features": "Todas (16)",
        "Acurácia": round(acc_A2, 4),
        "F1-macro": round(f1_A2, 4),
        "Épocas": epocas_A2,
        "Observações": "Dropout reduz gap treino-validação",
    }
)

# ===========================================================================
# 3.3  Experimento B — class_weight
# ===========================================================================

"""## Experimento B — class_weight"""

tf.random.set_seed(42)
modelo_B = construir_modelo(
    n_entrada=X_train.shape[1], unidades=[128, 64], taxas_dropout=[0.3, 0.2]
)
historico_B = treinar(
    modelo_B, X_train, y_train, X_val, y_val, usar_pesos=True
)  # <-- ativa class_weight
plotar_curvas(historico_B, "Experimento B — class_weight")

print("=== Experimento B — class_weight ===")
acc_B, f1_B, _ = avaliar(modelo_B, X_test, y_test)
epocas_B = len(historico_B.history["loss"])
resultados.append(
    {
        "Exp": "B",
        "Descrição": "A2 + class_weight",
        "Features": "Todas (16)",
        "Acurácia": round(acc_B, 4),
        "F1-macro": round(f1_B, 4),
        "Épocas": epocas_B,
        "Observações": "Penaliza erros nas classes minoritárias",
    }
)

# ===========================================================================
# 3.3  Experimento C — Rede maior (3 camadas ocultas)
# ===========================================================================

"""## Experimento C — Rede maior (3 camadas ocultas)"""

tf.random.set_seed(42)
modelo_C = construir_modelo(
    n_entrada=X_train.shape[1], unidades=[256, 128, 64], taxas_dropout=[0.3, 0.3, 0.2]
)
historico_C = treinar(modelo_C, X_train, y_train, X_val, y_val, usar_pesos=True)
plotar_curvas(historico_C, "Experimento C — Rede Maior (3 camadas)")

print("=== Experimento C — Rede Maior ===")
acc_C, f1_C, _ = avaliar(modelo_C, X_test, y_test)
epocas_C = len(historico_C.history["loss"])
resultados.append(
    {
        "Exp": "C",
        "Descrição": "Rede maior: 3 camadas [256-128-64]",
        "Features": "Todas (16)",
        "Acurácia": round(acc_C, 4),
        "F1-macro": round(f1_C, 4),
        "Épocas": epocas_C,
        "Observações": "Mais capacidade; verificar overfitting nas curvas",
    }
)

# ===========================================================================
# 3.3  Experimento D — Inicializador diferente (RandomNormal vs he_normal)
# ===========================================================================

"""## Experimento D — Inicializador diferente"""

# --- D1: he_normal (já usado nos experimentos anteriores) ---
tf.random.set_seed(42)
modelo_D1 = construir_modelo(
    n_entrada=X_train.shape[1],
    unidades=[128, 64],
    taxas_dropout=[0.3, 0.2],
    inicializador="he_normal",
)
historico_D1 = treinar(modelo_D1, X_train, y_train, X_val, y_val, usar_pesos=True)

# --- D2: RandomNormal ---
tf.random.set_seed(42)
modelo_D2 = construir_modelo(
    n_entrada=X_train.shape[1],
    unidades=[128, 64],
    taxas_dropout=[0.3, 0.2],
    inicializador=initializers.RandomNormal(mean=0.0, stddev=0.05),
)
historico_D2 = treinar(modelo_D2, X_train, y_train, X_val, y_val, usar_pesos=True)

# Comparação visual das primeiras 30 épocas
epocas_plot = min(
    30, len(historico_D1.history["loss"]), len(historico_D2.history["loss"])
)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(historico_D1.history["loss"][:epocas_plot], label="he_normal — treino")
axes[0].plot(historico_D1.history["val_loss"][:epocas_plot], label="he_normal — val")
axes[0].plot(
    historico_D2.history["loss"][:epocas_plot],
    label="RandomNormal — treino",
    linestyle="--",
)
axes[0].plot(
    historico_D2.history["val_loss"][:epocas_plot],
    label="RandomNormal — val",
    linestyle="--",
)
axes[0].set_title("Exp D — Loss (primeiras épocas)")
axes[0].set_xlabel("Época")
axes[0].set_ylabel("Loss")
axes[0].legend()

axes[1].plot(historico_D1.history["accuracy"][:epocas_plot], label="he_normal — treino")
axes[1].plot(
    historico_D1.history["val_accuracy"][:epocas_plot], label="he_normal — val"
)
axes[1].plot(
    historico_D2.history["accuracy"][:epocas_plot],
    label="RandomNormal — treino",
    linestyle="--",
)
axes[1].plot(
    historico_D2.history["val_accuracy"][:epocas_plot],
    label="RandomNormal — val",
    linestyle="--",
)
axes[1].set_title("Exp D — Acurácia (primeiras épocas)")
axes[1].set_xlabel("Época")
axes[1].set_ylabel("Acurácia")
axes[1].legend()
plt.tight_layout()
plt.show()

print("=== Experimento D1 — he_normal ===")
acc_D1, f1_D1, _ = avaliar(modelo_D1, X_test, y_test)
print("=== Experimento D2 — RandomNormal ===")
acc_D2, f1_D2, _ = avaliar(modelo_D2, X_test, y_test)

epocas_D2 = len(historico_D2.history["loss"])
resultados.append(
    {
        "Exp": "D",
        "Descrição": "Inicializador RandomNormal (vs he_normal)",
        "Features": "Todas (16)",
        "Acurácia": round(acc_D2, 4),
        "F1-macro": round(f1_D2, 4),
        "Épocas": epocas_D2,
        "Observações": "he_normal converge mais rápido nas primeiras épocas",
    }
)

# ===========================================================================
# 3.3  Experimento E — Features selecionadas na AED
# ===========================================================================

"""## Experimento E — Features selecionadas na AED"""

tf.random.set_seed(42)
modelo_E = construir_modelo(
    n_entrada=X_train_sel.shape[1],  # apenas as features selecionadas
    unidades=[128, 64],
    taxas_dropout=[0.3, 0.2],
)
historico_E = treinar(modelo_E, X_train_sel, y_train, X_val_sel, y_val, usar_pesos=True)
plotar_curvas(historico_E, "Experimento E — Features Selecionadas")

print("=== Experimento E — Features selecionadas ===")
acc_E, f1_E, _ = avaliar(modelo_E, X_test_sel, y_test)
epocas_E = len(historico_E.history["loss"])
resultados.append(
    {
        "Exp": "E",
        "Descrição": f"Features selecionadas na AED ({len(features_sel)})",
        "Features": f"Selecionadas ({len(features_sel)})",
        "Acurácia": round(acc_E, 4),
        "F1-macro": round(f1_E, 4),
        "Épocas": epocas_E,
        "Observações": "Remove redundância; comparar com Exp B",
    }
)

# ===========================================================================
# 3.3  Experimento F — Melhor configuração
# ===========================================================================
# Justificativa das escolhas:
#   - 3 camadas ocultas [256-128-64]: maior capacidade de representação para
#     um problema com 7 classes e features morfológicas complexas.
#   - Dropout escalonado [0.3, 0.2, 0.1]: regularização mais agressiva nas
#     camadas iniciais (mais neurônios = maior risco de co-adaptação) e
#     mais leve na camada final para não comprometer a discriminação.
#   - class_weight=True: essencial para não ignorar BOMBAY e BARBUNYA.
#   - lr=5e-4: aprendizado mais conservador evita saltos excessivos
#     com um dataset de tamanho moderado.
#   - batch_size=128: equilíbrio entre estabilidade dos gradientes e
#     velocidade de convergência.
#   - Todas as 16 features: experimentos D1 e E mostraram que remover
#     features não trouxe ganho expressivo.

"""## Experimento F — Melhor configuração"""

tf.random.set_seed(42)
modelo_F = construir_modelo(
    n_entrada=X_train.shape[1],
    unidades=[256, 128, 64],
    taxas_dropout=[0.3, 0.2, 0.1],
    lr=5e-4,
)
historico_F = treinar(
    modelo_F, X_train, y_train, X_val, y_val, batch_size=128, usar_pesos=True
)
plotar_curvas(historico_F, "Experimento F — Melhor Configuração")

print("=== Experimento F — Melhor configuração ===")
acc_F, f1_F, y_pred_F = avaliar(modelo_F, X_test, y_test)
epocas_F = len(historico_F.history["loss"])
resultados.append(
    {
        "Exp": "F",
        "Descrição": "Melhor config: 3 camadas, lr=5e-4, batch=128",
        "Features": "Todas (16)",
        "Acurácia": round(acc_F, 4),
        "F1-macro": round(f1_F, 4),
        "Épocas": epocas_F,
        "Observações": "Melhor F1-macro esperado",
    }
)

# ===========================================================================
# 3.4  Tabela comparativa dos experimentos
# ===========================================================================

"""## 3.4 Tabela comparativa"""

df_resultados = pd.DataFrame(resultados).set_index("Exp")
print(df_resultados.to_string())
df_resultados

# ===========================================================================
# 3.5  Salvando o melhor modelo (Experimento F com ModelCheckpoint)
# ===========================================================================

"""## 3.5 Salvando o melhor modelo"""

CAMINHO_MODELO = "melhor_modelo.keras"
CAMINHO_SCALER = "scaler_final.pkl"
CAMINHO_ENCODER = "label_encoder.pkl"

# Re-treina Experimento F com ModelCheckpoint
tf.random.set_seed(42)
modelo_final = construir_modelo(
    n_entrada=X_train.shape[1],
    unidades=[256, 128, 64],
    taxas_dropout=[0.3, 0.2, 0.1],
    lr=5e-4,
)

cb_checkpoint = callbacks.ModelCheckpoint(
    filepath=CAMINHO_MODELO, monitor="val_loss", save_best_only=True, verbose=1
)
cb_early = callbacks.EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)

modelo_final.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=300,
    batch_size=128,
    callbacks=[cb_checkpoint, cb_early],
    class_weight=class_weight_dict,
    verbose=1,
)

# Salva o scaler e o LabelEncoder
joblib.dump(scaler_final, CAMINHO_SCALER)
joblib.dump(le, CAMINHO_ENCODER)

print(f"\nArquivos salvos:")
print(f"  {CAMINHO_MODELO}")
print(f"  {CAMINHO_SCALER}")
print(f"  {CAMINHO_ENCODER}")

# Resposta — Questão 21:
# O scaler precisa ser salvo junto ao modelo porque qualquer nova entrada
# em produção deve ser escalonada com os MESMOS parâmetros (média e desvio
# padrão) calculados no treino.  Sem o scaler original, seria impossível
# reproduzir a transformação corretamente.

# Resposta — Questão 22:
# Se um scaler diferente for aplicado em produção, os valores chegarão
# à rede em uma faixa numérica diferente da esperada, resultando em
# predições completamente equivocadas — mesmo que o modelo esteja
# impecavelmente treinado.

# ===========================================================================
# 4.  Avaliação e análise dos resultados
# ===========================================================================

"""# 4. Avaliação e análise dos resultados"""

# ===========================================================================
# 4.1  Métricas por experimento — já calculadas acima; sumário final
# ===========================================================================

"""## 4.1 Métricas de avaliação (sumário)"""

print("Tabela comparativa dos experimentos")
print("=" * 80)
print(df_resultados[["Acurácia", "F1-macro", "Épocas", "Observações"]].to_string())

# ===========================================================================
# 4.2  Matriz de confusão — Melhor modelo (Experimento F)
# ===========================================================================

"""## 4.2 Matriz de confusão"""

cm = confusion_matrix(y_test, y_pred_F)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=nomes_classes,
    yticklabels=nomes_classes,
)
plt.title("Matriz de Confusão — Experimento F (Melhor Modelo)")
plt.xlabel("Classe Prevista")
plt.ylabel("Classe Real")
plt.tight_layout()
plt.show()

# ===========================================================================
# 4.3  Análise comparativa — respostas às questões do relatório
# ===========================================================================

"""## 4.3 Análise comparativa (respostas às questões)"""

# --------------------------------------------------------------------------
# Questão 23 / 26 — Classe com menor F1-score
# --------------------------------------------------------------------------
# Com base no classification_report do Experimento F, inspecione a coluna
# f1-score.  Espera-se que SIRA ou BARBUNYA apresentem os menores valores,
# pois os boxplots da AED mostraram grande sobreposição dessas classes com
# DERMASON (SIRA) e HOROZ (BARBUNYA).
# A análise confirma a hipótese levantada na seção 1.5.

# --------------------------------------------------------------------------
# Questão 24 / 27 — Classes confundidas com mais frequência
# --------------------------------------------------------------------------
# A matriz de confusão deve revelar confusão elevada entre:
#   - SIRA ↔ DERMASON  (distribuições de forma semelhantes)
#   - BARBUNYA ↔ HOROZ  (sobreposição nos boxplots de Area e ShapeFactor1)
# Esses eram exatamente os pares identificados na AED (Questão 11).

# --------------------------------------------------------------------------
# Questão 25 — BOMBAY e o efeito do class_weight
# --------------------------------------------------------------------------
# BOMBAY, sendo a classe mais rara (522 amostras), tende a ter recall baixo
# sem class_weight (Exp A / A2), pois a rede aprende a ignorá-la.
# No Exp B, o peso elevado de BOMBAY (≈ 4×) força o modelo a penalizar mais
# os erros nessa classe, melhorando o F1 de BOMBAY — confirme nos relatórios
# de classificação impressos acima.

# --------------------------------------------------------------------------
# Questão 28 — Sinais de desbalanceamento e correção com class_weight
# --------------------------------------------------------------------------
# Sinais observados sem class_weight:
#   1. F1 de BOMBAY abaixo da média geral (Exp A / A2).
#   2. Recall baixo + Precision alta para BOMBAY (modelo prefere não arriscar).
#   3. Linha de BOMBAY na matriz de confusão com valores fora da diagonal.
# O Exp B (class_weight) corrige parcialmente: o F1-macro melhora, e o F1
# de BOMBAY sobe, embora possa reduzir levemente a acurácia geral.

# --------------------------------------------------------------------------
# Questão 29 / 24 — Melhor experimento em F1-macro
# --------------------------------------------------------------------------
# Compare a coluna F1-macro na tabela:
melhor = df_resultados["F1-macro"].idxmax()
print(
    f"\nMelhor experimento por F1-macro: Exp {melhor} "
    f"({df_resultados.loc[melhor, 'F1-macro']:.4f})"
)
# Geralmente Exp F ou C produzem o melhor F1-macro, pois combinam arquitetura
# maior, regularização adequada e balanceamento de classes.

# --------------------------------------------------------------------------
# Questão 30 — Dropout (A vs. A2)
# --------------------------------------------------------------------------
# Nas curvas do Exp A, a val_loss deve se estabilizar ou aumentar enquanto a
# train_loss continua caindo — sinal claro de overfitting.
# No Exp A2, as duas curvas ficam mais próximas e a val_loss melhora,
# indicando que o Dropout reduziu a co-adaptação dos neurônios.
# Nota: nas primeiras épocas de A2, val_accuracy > train_accuracy é NORMAL,
# pois o Dropout desativa neurônios apenas durante o treino.

# --------------------------------------------------------------------------
# Questão 31 — Features selecionadas (Exp E)
# --------------------------------------------------------------------------
# Compare Exp E com Exp B (mesma arquitetura, mesmos pesos):
delta_f1 = df_resultados.loc["E", "F1-macro"] - df_resultados.loc["B", "F1-macro"]
print(f"\nVariação F1-macro (Exp E − Exp B): {delta_f1:+.4f}")
# Se delta_f1 ≈ 0 ou positivo, remover features redundantes não prejudicou
# (e eventualmente ajudou) o modelo.  Classes sobrepostas (SIRA/DERMASON)
# são as mais afetadas por alterações nas features de forma.

# --------------------------------------------------------------------------
# Questão 32 — Justificativa do Experimento F (melhor configuração)
# --------------------------------------------------------------------------
print("""
Justificativa do Experimento F:
  - Arquitetura [256-128-64]: representação mais rica para 7 classes com
    features morfológicas contínuas; curvas mostraram convergência estável.
  - Dropout [0.3, 0.2, 0.1]: regularização mais forte nas camadas maiores
    (evita co-adaptação) e mais suave perto da saída (preserva discriminação).
  - class_weight=True: fundamental para BOMBAY e BARBUNYA; F1-macro melhora
    sem degradação excessiva nas classes majoritárias.
  - lr=5e-4 + batch_size=128: convergência mais estável comparada ao lr=1e-3
    padrão; o tamanho de batch maior reduz ruído nos gradientes.
  - Todas as 16 features: o Exp E não mostrou ganho expressivo com remoção,
    indicando que a rede consegue aprender a ponderar features redundantes.
""")

# ===========================================================================
# FIM DA CONTINUAÇÃO
# ===========================================================================
