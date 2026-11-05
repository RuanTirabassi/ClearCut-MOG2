# ClearCut-MOG2

**Removedor de fundo em tempo real para webcam e vídeos, feito só com visão computacional clássica, sem redes neurais.**

O programa separa a pessoa do cenário e troca o fundo por outra imagem, ao vivo. Mostra lado a lado o quadro original, a máscara (branco = pessoa) e o resultado com o fundo novo, além de contar quantos objetos (pessoas) há na cena.

Projeto da disciplina de Visão Computacional (UENP), inspirado no [ClearCut](https://github.com/KrishBharadwaj5678/ClearCut). O original usa o SelfieSegmentation do MediaPipe, uma rede neural. Aqui a experiência é a mesma (webcam, pasta de fundos, troca com A/D), mas a segmentação foi refeita com **subtração de fundo (MOG2)** e técnicas clássicas do OpenCV.

> Por privacidade (LGPD), os vídeos e imagens de teste do grupo não estão no repositório. Para testar, use a sua webcam ou grave um vídeo seu.

---

## Sumário

- [Como funciona](#como-funciona)
- [Tecnologias](#tecnologias)
- [Instalação](#instalação)
- [Como usar](#como-usar)
- [Dicas para um bom resultado](#dicas-para-um-bom-resultado)
- [Parâmetros](#parâmetros)
- [Avaliação (IoU)](#avaliação-iou)
- [Resultados](#resultados)
- [Limitações](#limitações)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Referências](#referências)

---

## Como funciona

### A ideia: subtração de fundo

Com a câmera parada, o programa primeiro **aprende como é a cena vazia**. Depois, todo pixel que ficar muito diferente do que foi aprendido é considerado **pessoa**.

Para isso usamos o **MOG2** (*Mixture of Gaussians*, `cv2.createBackgroundSubtractorMOG2`). Cada pixel guarda até 5 curvas gaussianas (em forma de sino), e cada curva tem três números:

| Símbolo | Nome | Significado |
|---|---|---|
| μ | média | a "cor de costume" do pixel |
| σ² | variância | o quanto essa cor costuma variar (ruído, tremida da luz) |
| π | peso | o quanto essa curva aparece no histórico do pixel |

Para cada cor nova `x` (canais B, G e R), o MOG2 calcula a distância até a média, dividida pela variância:

```
D² = [(xB − μB)² + (xG − μG)² + (xR − μR)²] / σ²
```

Se `D² < T` (o `varThreshold`, aqui 40), a cor é "de costume" e o pixel é **fundo**. Se não, é **pessoa** (255). Quando a cor é a mesma do fundo, só que mais escura, o MOG2 marca como **sombra** (127).

O programa funciona em duas fases:

1. **Aprende** o fundo com a cena vazia nos primeiros 60 quadros (`learningRate = -1`, automático).
2. **Congela** o modelo (`learningRate = 0`). Assim, uma pessoa parada não vai sendo "absorvida" pelo fundo.

### Pipeline

```
quadro (webcam ou vídeo, redimensionado para altura 480)
 → 0. Compensação de exposição   1 ganho por canal de cor = valor mais comum (moda) de
                                 fundo_aprendido / quadro; desfaz o brilho que a câmera
                                 muda sozinha quando a pessoa entra
 → 1. Filtro gaussiano 5×5       reduz o ruído do sensor
 → 2. MOG2                       fase 1 aprende (60 quadros), fase 2 congelado
 → 3. Limiar em 200              descarta a sombra (127), fica só a pessoa (255)
 → 4. Abertura + fechamento      elipse 7×7: apaga pontinhos soltos e tapa furinhos
 → 5. Contornos                  só manchas com área ≥ 2000 px, preenchidas;
                                 nº de contornos = nº de objetos
 → 6. Buracos grandes reabertos  buraco > 1500 px (ex.: vão entre braço e corpo) volta a ser fundo
 → 7. Composição                 máscara encolhida (erosão 5) e suavizada (blur 9) → alfa
                                 resultado = alfa · quadro + (1 − alfa) · fundo novo
 → painel: original | máscara | resultado  (+ nome do fundo e nº de objetos)
```

### Exposição automática

O maior desafio foi a câmera, não o MOG2. Quando a pessoa entra, a câmera ajusta o brilho sozinha. A parede inteira muda de cor e o modelo passa a achar que tudo é pessoa: no vídeo claro, 92% da imagem foi marcada. Duas soluções:

- **Compensação (passo 0):** multiplica cada canal por um ganho que devolve o quadro ao brilho do fundo aprendido. O ganho é a **moda**, e não a média ou a mediana: todos os pixels do fundo pedem o mesmo ganho e formam um pico no histograma, então funciona mesmo com a pessoa ocupando metade da imagem.
- **Trava (só webcam):** depois da contagem, o programa desliga a exposição automática pelo OpenCV (`CAP_PROP_AUTO_EXPOSURE`) e escolhe o valor manual com brilho mais parecido com o automático. Ao reaprender (B) ou sair, a câmera volta ao automático.

---

## Tecnologias

| Tecnologia | Uso |
|---|---|
| **Python** 3.10–3.13 | linguagem |
| **OpenCV** 4.10 (`opencv-python`) | captura da webcam/vídeo, MOG2, filtros, morfologia, contornos, janela e gravação |
| **NumPy** 2.2 | operações com as imagens (máscaras, ganhos, histograma, composição alfa) |

Não usa redes neurais, GPU nem modelos pré-treinados.

---

## Instalação

```bash
git clone <url-deste-repositório>
cd ClearCut-MOG2
pip install -r requirements.txt
```

Se você já tiver outra versão do OpenCV instalada, remova antes para evitar conflito:

```bash
pip uninstall opencv-python opencv-contrib-python
```

---

## Como usar

Sempre rode **a partir da pasta do projeto**:

```bash
python codigo/clearcut_mog2.py                         # webcam
python codigo/clearcut_mog2.py videos/meu_video.mp4    # arquivo de vídeo
python codigo/clearcut_mog2.py 0 --gravar              # webcam + grava o painel em resultados/demo_webcam.mp4
```

**Na webcam:**

1. O programa conta 5 segundos (**"Saia da cena: 5, 4, 3..."**). Saia da frente da câmera.
2. Ele aprende o fundo por cerca de 2 segundos.
3. Quando o terminal mostrar **"Fundo aprendido! Pode entrar na cena."**, volte.

**Em vídeo:** os primeiros 2–3 segundos do vídeo precisam ser de cena vazia. O programa salva sozinho 1 quadro a cada 20 em `resultados/` (frame, máscara e resultado).

### Teclas

| Tecla | Ação |
|---|---|
| `A` / `D` | fundo anterior / próximo (imagens da pasta `fundos/`) |
| `B` | reaprender o fundo (saia da cena antes; na webcam a contagem recomeça) |
| `S` | salvar quadro, máscara e resultado em `resultados/` |
| `Q` | sair (fechar no X da janela também funciona) |

Para usar seus próprios fundos, coloque imagens `.jpg` ou `.png` na pasta `fundos/`.

---

## Dicas para um bom resultado

- **Câmera parada.** Se ela mexer, tudo vira "pessoa".
- **Cena vazia** enquanto o fundo é aprendido. Se você estiver na frente, vira parte do fundo.
- **Fundo estável:** nada se mexendo atrás (cortina, ventilador ligado, TV).
- **Luz constante:** não acender ou apagar a luz depois de aprender o fundo.
- **Roupa que contraste com o fundo.** Camiseta escura na frente de um objeto escuro "some" (camuflagem).
- **Gravando com o celular:** apoie o aparelho na horizontal, trave a exposição se der (segure o dedo na tela até aparecer "AE/AF travado") e passe o vídeo para o PC por cabo ou nuvem, não por WhatsApp, que comprime o vídeo.

---

## Parâmetros

Todos ficam no topo de `codigo/clearcut_mog2.py`, comentados.

| Constante | Valor | Aumentar / diminuir |
|---|---|---|
| `BLUR` | 5 | ↑ menos ruído, bordas mais borradas |
| `HISTORY` | 200 | memória do MOG2; pesa pouco porque o modelo é congelado |
| `VAR_THRESHOLD` | 40 | ↓ mais sensível (pega mais da pessoa e mais ruído); ↑ menos ruído e mais buracos |
| `FRAMES_APRENDER` | 60 | ↓ fundo mal aprendido; ↑ mais tempo com a cena vazia |
| `ESPERA_WEBCAM` | 5 | segundos de contagem antes de aprender o fundo (webcam) |
| `LIMIAR` | 200 | abaixo de 127 a sombra passa a contar como pessoa |
| `KERNEL` | 7 | ↑ limpa mais, mas engorda a silhueta e apaga partes finas |
| `AREA_MIN` | 2000 | ↑ ignora mais manchas, mas pode apagar uma mão separada do corpo |
| `BURACO_MAX` | 1500 | ↑ o vão entre braço e corpo volta a mostrar a parede antiga; ↓ reabre falhas reais na pessoa |
| `TRAVAR_EXPOSICAO` | True | só webcam; False = a câmera clareia sozinha e o fundo some |
| `COMPENSAR_EXPOSICAO` | True | False = sem correção de brilho |
| `ENCOLHER_BORDA` | 5 | ↑ menos "halo" da parede antiga em volta da pessoa, mas come cabelo e dedos |
| `SUAVIZAR_BORDA` | 9 | ↑ transição mais suave (e mais borrada) entre pessoa e fundo |

---

## Avaliação (IoU)

Para medir a qualidade, comparamos a máscara do programa com uma máscara correta pintada à mão (o **gabarito**).

```bash
python codigo/criar_gabarito.py                  # pinta os gabaritos dos frames salvos em resultados/
python codigo/criar_gabarito.py claro_0520       # abre só os frames cujo nome contém esse trecho
python codigo/avaliacao.py                       # calcula o IoU e gera resultados/tabela_acertos.csv
```

**No `criar_gabarito.py`:**

| Comando | Ação |
|---|---|
| botão esquerdo | pinta a pessoa |
| botão direito | apaga |
| `+` / `-` | muda o tamanho do pincel |
| `S` | salva em `gabarito/` |
| `N` | pula |
| `Q` | sai |

Se o frame já tiver gabarito, ele abre o salvo para corrigir.

**IoU** (*Intersection over Union*) = pixels de pessoa em comum ÷ pixels de pessoa no programa **ou** no gabarito. 1 = máscaras iguais; 0 = nada em comum. Contamos **acerto** quando IoU ≥ 0,8.

---

## Resultados

Testado em dois vídeos gravados pelo grupo (um claro, outro com pouca luz) e na webcam do notebook.

| Medida | Resultado |
|---|---|
| IoU médio (12 frames com a pessoa) | 0,791 |
| Acertos (IoU ≥ 0,8) | 4 de 12 (33,3%) |
| Contagem de objetos correta | 75 de 78 frames (96,2%) |
| Falso positivo (fundo marcado como pessoa) | no máximo 2,6% |
| Velocidade na webcam | 27–30 ms por quadro (~35 fps) |

- Os frames que não passaram do corte ficaram entre 0,74 e 0,79. O erro principal é a **camuflagem**: a camiseta escura passa na frente de um ventilador preto e parte do corpo some (12–26% da pessoa).
- Baixar o `VAR_THRESHOLD` para 16 melhora os vídeos (8 de 12 acertos). Na webcam, porém, aparecem manchas falsas: 128 de 603 quadros com mais de 1 objeto, contra 14 com o valor 40. Por isso o padrão ficou em 40.
- No vídeo claro, sem a compensação de exposição, 92% da imagem era marcada como pessoa. Com ela, a área marcada caiu para 18–26%.

---

## Limitações

- Câmera **parada** e cena **vazia** no início.
- **Luz estável:** o modelo é congelado e não se adapta a mudanças grandes de iluminação.
- **Camuflagem:** partes da pessoa da mesma cor do fundo somem.
- **Duas pessoas encostadas** viram um contorno só, e a contagem erra.

---

## Estrutura do projeto

```
ClearCut-MOG2/
├── codigo/
│   ├── clearcut_mog2.py    programa principal (pipeline, webcam/vídeo, teclas, gravação)
│   ├── criar_gabarito.py   ferramenta para pintar as máscaras corretas à mão
│   └── avaliacao.py        calcula o IoU e gera a tabela de acertos (CSV)
├── fundos/                 imagens de fundo para trocar com A/D
├── relatorio/              relatório em LaTeX (main.tex, capitulos/, referencias.bib)
├── requirements.txt
└── README.md

Criadas ao rodar (fora do git):
├── videos/                 seus vídeos de teste
├── resultados/             frame_, mascara_, resultado_*.png, demo_*.mp4, tabela_acertos.csv
└── gabarito/               gabarito_*.png
```

As figuras do relatório mostram um integrante do grupo e não estão no repositório. Por isso o LaTeX não compila só com o que está aqui.

---

## Referências

- ZIVKOVIC, Z. *Improved adaptive Gaussian mixture model for background subtraction*. ICPR, 2004.
- ZIVKOVIC, Z.; VAN DER HEIJDEN, F. *Efficient adaptive density estimation per image pixel for the task of background subtraction*. Pattern Recognition Letters, 2006.
- OpenCV: [How to Use Background Subtraction Methods](https://docs.opencv.org/4.10.0/d1/dc5/tutorial_background_subtraction.html) e [`cv::BackgroundSubtractorMOG2`](https://docs.opencv.org/4.10.0/d7/d7b/classcv_1_1BackgroundSubtractorMOG2.html).
- GONZALEZ, R. C.; WOODS, R. E. *Processamento Digital de Imagens*. 3. ed. Pearson, 2010.
- Projeto original: [KrishBharadwaj5678/ClearCut](https://github.com/KrishBharadwaj5678/ClearCut).

Trabalho da Avaliação 2 de Visão Computacional, UENP, Prof. Tafarel Rocha da Silva.
