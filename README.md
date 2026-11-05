# ClearCut-MOG2 — Removedor de fundo em tempo real (sem redes neurais)

Trabalho da Avaliação 2 de Visão Computacional.

Inspirado no **ClearCut** (<https://github.com/KrishBharadwaj5678/ClearCut>), que troca o fundo da webcam com as teclas A/D usando o SelfieSegmentation do MediaPipe (uma rede neural). Como a disciplina não permite redes neurais neste trabalho, mantivemos a mesma experiência (webcam, pasta de fundos, teclas A/D) e trocamos a segmentação por técnicas clássicas de OpenCV.

**Conceito novo:** subtração de fundo com `cv2.createBackgroundSubtractorMOG2`.

> **Privacidade (LGPD):** este repositório **não contém** os vídeos, frames, máscaras, gabaritos, figuras do relatório nem os slides, porque todos mostram um integrante do grupo. Só sobem o código, os fundos de exemplo e o texto do relatório (veja o [`.gitignore`](.gitignore)). Para testar, use a webcam ou grave um vídeo seu (seção [Como gravar os vídeos](#como-gravar-os-vídeos)). As pastas `resultados/` e `gabarito/` são criadas sozinhas pelos programas; a pasta `videos/` você cria.

## Pipeline

```
frame (webcam ou vídeo, altura 480)
   (webcam: a exposição automática da câmera é TRAVADA quando o fundo começa a ser aprendido)
 → 0. Compensação de exposição     desfaz o brilho/cor que a câmera muda sozinha
                                   (1 ganho por cor = valor MAIS COMUM (moda) de
                                    fundo_aprendido / frame: todo o fundo pede o mesmo ganho)
 → 1. GaussianBlur 5x5             reduz o ruído do sensor
 → 2. MOG2                         fase 1: aprende o fundo com a cena vazia (60 frames)
                                   fase 2: learningRate=0 (modelo congelado)
 → 3. Limiar em 200                remove a sombra (o MOG2 marca sombra com 127)
 → 4. Abertura + fechamento        tira pontinhos de ruído e fecha buracos
 → 5. Contornos + área mínima      fica só o que é grande, preenchido → máscara
                                   nº de contornos = nº de objetos/pessoas
 → 6. Buracos grandes reabertos    vão entre braço e corpo (> 1500 px) volta a ser fundo
 → 7. Composição com borda suave   máscara encolhida 2 px e borrada → alfa
                                   resultado = alfa·frame + (1−alfa)·fundo escolhido
 → painel: original | máscara | resultado  (+ nome do fundo e nº de objetos)
```

## Instalação

Python 3.10 a 3.13. Na pasta do projeto:

```bash
pip install -r requirements.txt
```

(instala `opencv-python==4.10.0.84` e `numpy==2.2.6`. Se tiver outra versão do OpenCV, desinstale antes com `pip uninstall opencv-python opencv-contrib-python`.)

## Como rodar

Sempre a partir da **pasta do projeto**:

```bash
python codigo/clearcut_mog2.py                          # webcam
python codigo/clearcut_mog2.py videos/video_claro.mp4   # vídeo
python codigo/clearcut_mog2.py videos/video_claro.mp4 --gravar    # + grava resultados/demo_<video>.mp4
python codigo/criar_gabarito.py                         # pintar os gabaritos (ou só alguns: ver abaixo)
python codigo/avaliacao.py                              # tabela de acertos
```

Enquanto aparece **"Aprendendo fundo..."** a cena precisa estar **vazia**. Se a pessoa estiver na frente, ela vira parte do "fundo" e depois só aparecem os **contornos** de onde ela se mexeu.

**Na webcam**, o programa faz antes uma contagem de 5 s (**"Saia da cena: 5, 4, 3..."**, constante `ESPERA_WEBCAM`): saia da frente da câmera, espere aparecer **"Fundo aprendido! Pode entrar na cena."** no terminal (~2 s depois) e volte. Para refazer, aperte **B** (a contagem recomeça).

**Exposição travada (webcam):** quando a contagem termina, o programa desliga a exposição automática da webcam. A imagem pode piscar por ~1 s enquanto ele escolhe o valor com o mesmo brilho. Sem isso, quando a pessoa chega perto da câmera, a webcam clareia a imagem toda (no nosso teste a parede foi de ~130 para ~235) e o fundo "some". No **B** a câmera volta ao automático para se reajustar durante a contagem, e ao sair (Q ou X da janela) ela é devolvida no automático para os outros programas. Se a sua webcam não aceitar, o terminal avisa e fica só a compensação.

## Teclas

| Tecla | `clearcut_mog2.py` |
|---|---|
| `A` / `D` | fundo anterior / próximo (imagens da pasta `fundos/`) |
| `B` | reaprender o fundo (saia da cena; na webcam há contagem de 5 s) |
| `S` | salvar frame, máscara e resultado em `resultados/` |
| `Q` | sair (fechar no X da janela também funciona e fecha o vídeo gravado direito) |

Nos vídeos, o programa também salva sozinho 1 frame a cada 20 (`SALVAR_A_CADA`), para a avaliação.

## Estrutura de pastas

Marcadas com `*`: só existem no PC e no .zip de entrega, **não vão para o GitHub** (LGPD).

```
projeto/
├── codigo/
│   ├── clearcut_mog2.py    programa principal
│   ├── criar_gabarito.py   pinta as máscaras corretas à mão
│   └── avaliacao.py        IoU, tabela de acertos e CSV
├── fundos/                 imagens de fundo (coloque fotos aqui; .jpg ou .png)
├── videos/  *              video_claro.mp4, video_escuro.mp4
├── resultados/  *          frame_, mascara_, resultado_*.png, demo_*.mp4, tabela_acertos.csv
├── gabarito/  *            gabarito_*.png (feitos à mão)
├── relatorio/              relatório em LaTeX (main.tex, capitulos/, referencias.bib)
│   └── figuras/            só o logo sobe; as figuras com a pessoa (fig_*.jpg) ficam fora *
├── LEIAME.txt              instruções curtas para a entrega
├── README.md
└── requirements.txt
```

Como as figuras `fig_*.jpg` não estão no repositório, o relatório não compila direto daqui: use o .zip do Overleaf do grupo.

## Resultados

Avaliação em 12 frames com a pessoa (6 de cada vídeo), comparando com gabaritos pintados à mão:

| Medida | Resultado |
|---|---|
| IoU médio | 0,791 |
| Acertos (IoU ≥ 0,8) | 4 de 12 (33,3%) |
| Contagem de objetos correta | 75 de 78 frames (96,2%) |
| Velocidade na webcam | 27–30 ms por frame (~35 fps) |

Os frames abaixo de 0,8 ficaram entre 0,74 e 0,79. O erro principal é a **camuflagem**: a camiseta escura passa na frente do ventilador preto e parte da pessoa some. Baixar o `VAR_THRESHOLD` para 16 sobe os vídeos para 8 de 12, mas na webcam cria manchas falsas (128 de 603 frames com mais de 1 objeto, contra 14 com 40), por isso mantivemos 40. Detalhes no relatório.

## Como gravar os vídeos

1. **Celular apoiado**, na **horizontal**. Depois de começar a gravar, ninguém encosta nele: se a câmera mexer, tudo vira "pessoa".
2. **Primeiros 2–3 s com a cena vazia.** É quando o programa aprende o fundo.
3. Depois a pessoa **entra, anda, fica parada uns segundos e sai**. Pelo menos **20 s** por vídeo.
4. **Fundo estável** (nada se mexendo atrás: cortina, ventilador, TV) e **luz uniforme** (não acender/apagar luz).
   Se der, **trave a exposição** da câmera (segure o dedo na tela até aparecer "AE/AF travado", ou use o modo Pro). Senão o celular clareia a imagem quando a pessoa entra; o programa compensa isso (`COMPENSAR_EXPOSICAO`), mas é melhor evitar.
5. Fique **a uma distância em que apareça o corpo inteiro ou da cintura para cima**, sem ocupar a tela toda.
6. Passe o vídeo para o PC **por cabo ou Google Drive**, não pelo WhatsApp (ele comprime e reduz a resolução).
7. Salve como `videos/video_claro.mp4` e `videos/video_escuro.mp4` (ou outro nome; o nome do vídeo vai no nome dos arquivos salvos).

## Casos difíceis para gravar de propósito

- **Roupa da cor do fundo** (ex.: camiseta branca na frente de parede branca): partes da pessoa somem (falso negativo). Na webcam, a camiseta escura "some" quando passa na frente do ventilador e da pilha de roupas escuras; para a demonstração, tire objetos escuros de trás.
- **Sombra forte** (luz lateral perto da pessoa): parte da sombra pode virar pessoa (falso positivo).
- **Luz diferente**: pouca luz, ou mudar a luz no meio do vídeo. O modelo congelado não se adapta.
- **Duas pessoas encostadas**: viram um contorno só (a contagem erra).
- **Pessoa parada por vários segundos**: deve continuar aparecendo, por causa do modelo congelado (é um caso de acerto que vale mostrar).

## Como fazer as máscaras gabarito

O gabarito é a "resposta certa": uma imagem do mesmo tamanho do frame, **branca onde está a pessoa e preta no resto**, feita à mão.

1. Rode o programa nos dois vídeos. Os frames ficam em `resultados/frame_<video>_<NNNN>.png`.
2. Rode `python codigo/criar_gabarito.py`. Ele abre, um por um, os frames que ainda não têm gabarito, ampliados na tela (`ZOOM = 1.5`; diminua se a janela não couber). Para abrir só alguns, passe pedaços do nome, por exemplo `python codigo/criar_gabarito.py claro_0520 escuro_0400`. Se esses já tiverem gabarito, ele abre o salvo para **corrigir**, sem começar do zero.
3. Pinte a pessoa com o **botão esquerdo** do mouse (aparece em vermelho transparente). O **botão direito** apaga. `+` e `-` mudam o tamanho do pincel: use pincel grande no meio do corpo e pequeno nas bordas. **A borda vermelha tem que parar exatamente na borda da pessoa.** Se passar para a parede, o programa é punido por um erro que é do gabarito.
4. `S` salva em `gabarito/gabarito_<video>_<NNNN>.png` e passa para o próximo frame. `N` pula (na correção, deixa como estava), `Q` sai (dá para continuar depois: ele só abre os que faltam).
5. Faça **pelo menos 10 frames** no total, dos dois vídeos, escolhendo frames **com a pessoa** visível. Os escolhidos para os nossos vídeos (fáceis e difíceis):

   ```bash
   python codigo/criar_gabarito.py claro_0520 claro_0600 claro_0760 claro_0820 claro_0880 claro_0940 escuro_0400 escuro_0460 escuro_0540 escuro_0580 escuro_0680 escuro_0720
   ```

**Alternativa (GIMP):** abra o frame, crie uma camada nova preenchida de preto com 50% de opacidade, pinte a pessoa de branco com o pincel, deixe a camada com 100% de opacidade, apague a camada do frame e exporte como `gabarito/gabarito_<mesmo final do frame>.png`.

## Avaliação

`python codigo/avaliacao.py` compara cada `mascara_*.png` com o `gabarito_*.png` do mesmo frame:

- **IoU** = pixels de pessoa em comum ÷ pixels de pessoa no programa **ou** no gabarito (1 = perfeito, 0 = nada em comum).
- **ACERTO** se IoU ≥ 0,8.
- Imprime a tabela (vídeo, frame, IoU, ACERTO/ERRO), a % de acertos por vídeo e no total, e salva `resultados/tabela_acertos.csv`.

## Parâmetros (topo de `codigo/clearcut_mog2.py`)

| Constante | Valor | Efeito de aumentar / diminuir |
|---|---|---|
| `BLUR` | 5 | ↑ menos ruído, bordas mais borradas |
| `HISTORY` | 200 | memória do MOG2; pesa pouco porque o modelo é congelado |
| `VAR_THRESHOLD` | 40 | ↓ mais sensível (pega mais da pessoa e mais ruído); ↑ menos ruído e mais buracos |
| `FRAMES_APRENDER` | 60 | ↓ fundo mal aprendido; ↑ mais tempo com a cena vazia |
| `LIMIAR` | 200 | abaixo de 127 a sombra passa a contar como pessoa |
| `KERNEL` | 7 | ↑ limpa mais, mas engorda a silhueta e apaga partes finas |
| `AREA_MIN` | 2000 | ↑ ignora mais manchas, mas pode apagar uma mão separada do corpo |
| `BURACO_MAX` | 1500 | buraco dentro da silhueta maior que isso volta a ser fundo (vão entre braço e corpo); ↑ o vão volta a mostrar a parede antiga, ↓ reabre falhas reais na pessoa |
| `TRAVAR_EXPOSICAO` | True | só webcam; False = a webcam clareia sozinha quando a pessoa chega perto e o fundo some |
| `COMPENSAR_EXPOSICAO` | True | False = sem correção de brilho; no nosso vídeo claro a parede inteira virava "pessoa" |
| `ENCOLHER_BORDA` | 5 | só na composição; ↑ menos "halo" da parede antiga em volta da pessoa, mas come cabelo e dedos |
| `SUAVIZAR_BORDA` | 9 | ↑ transição mais suave (e mais borrada) entre pessoa e fundo |
