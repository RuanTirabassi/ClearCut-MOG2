"""
ClearCut-MOG2: removedor de fundo em tempo real SEM redes neurais.

Inspirado no projeto ClearCut (https://github.com/KrishBharadwaj5678/ClearCut), que troca
o fundo da webcam com as teclas A/D usando o SelfieSegmentation (MediaPipe, rede neural).
Aqui mantemos a mesma ideia (webcam, pasta de fundos, teclas A/D), mas a segmentacao e
feita com tecnicas classicas de OpenCV + SUBTRACAO DE FUNDO (MOG2).

Uso (rodar a partir da pasta do projeto):
    python codigo/clearcut_mog2.py                    -> webcam
    python codigo/clearcut_mog2.py videos/video_claro.mp4  -> arquivo de video
    python codigo/clearcut_mog2.py 0 --gravar         -> webcam + grava o painel em resultados/

Teclas: A/D = fundo anterior/proximo | B = reaprender o fundo | S = salvar | Q = sair

IMPORTANTE: enquanto o fundo e aprendido a pessoa precisa estar FORA da cena. Se ela estiver
na frente, vira parte do "fundo" e depois so aparecem os contornos de onde ela se mexeu.
"""
import glob
import os
import sys
import time

import cv2
import numpy as np

# ============================ PARAMETROS ============================
ALTURA = 480             # altura de trabalho; a largura segue a proporcao do video.
                         # Maior = mais detalhe, porem mais lento.
BLUR = 5                 # kernel do GaussianBlur (impar). Maior = menos ruido, bordas mais borradas.
HISTORY = 200            # frames que o MOG2 considera no modelo. Maior = fundo mais estavel,
                         # porem demora mais para se adaptar (aqui pesa pouco, pois congelamos).
VAR_THRESHOLD = 40       # quao diferente do fundo o pixel precisa ser para virar "pessoa".
                         # Menor = mais sensivel (mais ruido); maior = menos ruido (mais buracos).
FRAMES_APRENDER = 60     # frames da fase 1 (cena vazia), ~2 s a 30 fps. Menos = fundo mal
                         # aprendido (mais ruido); mais = mais tempo esperando com a cena vazia.
ESPERA_WEBCAM = 5        # so na WEBCAM: segundos de contagem regressiva antes de aprender o fundo
                         # (ao abrir e ao apertar B), para dar tempo de sair da cena. Tambem deixa
                         # a camera terminar de ajustar o brilho. 0 = aprende na hora.
LIMIAR = 200             # o MOG2 marca sombra com 127 e pessoa com 255. Limiar 200 descarta a
                         # sombra. Abaixo de 127 a sombra passaria a contar como pessoa.
KERNEL = 7               # tamanho do kernel eliptico da morfologia. Maior = limpa mais ruido e
                         # fecha mais buracos, mas "engorda" a silhueta e apaga partes finas.
AREA_MIN = 2000          # area minima (pixels) de um contorno. Maior = ignora mais manchas,
                         # mas pode apagar partes reais (ex.: mao separada do corpo).
BURACO_MAX = 1500        # buraco DENTRO da silhueta maior que isto (pixels) volta a ser fundo: e o
                         # vao entre o braco e o corpo (mao no quadril). Menores sao falhas da
                         # deteccao e sao tapados. Maior = o vao volta a mostrar a parede antiga;
                         # menor = reabre falhas de verdade na pessoa.
TRAVAR_EXPOSICAO = True  # so na WEBCAM: desliga a exposicao automatica da camera enquanto o fundo
                         # aprendido vale (volta ao automatico no B e ao sair). False = a webcam
                         # clareia tudo quando a pessoa chega perto e o fundo "some".
COMPENSAR_EXPOSICAO = True  # corrige o brilho/cor que a camera muda sozinha quando a pessoa entra.
                         # False = sem correcao (no video claro e na webcam a parede virava pessoa).
ENCOLHER_BORDA = 5       # so na composicao: encolhe a mascara (erosao) antes de suavizar, para a
                         # transicao ficar DENTRO da pessoa e nao mostrar um "halo" da parede antiga
                         # em volta dela. Maior = menos halo, mas come cabelo e dedos. 1 = nao encolhe.
SUAVIZAR_BORDA = 9       # kernel (impar) do blur na mascara para a borda nao ficar "serrilhada".
                         # Maior = transicao mais suave (e mais borrada); 1 = borda dura.
SALVAR_A_CADA = 20       # em VIDEO, salva automaticamente a cada N frames (0 = so com a tecla S).
ESCALA_TELA = 0.7        # escala do painel mostrado na tela (so visual, nao afeta o resultado)
# ====================================================================

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # pasta do projeto
PASTA_FUNDOS = os.path.join(RAIZ, "fundos")
PASTA_RESULTADOS = os.path.join(RAIZ, "resultados")
FONTE = cv2.FONT_HERSHEY_SIMPLEX
JANELA = "ClearCut-MOG2  (original | mascara | resultado)"


def criar_subtrator():
    """Cria um MOG2 novo. detectShadows=True faz ele marcar sombras com 127."""
    return cv2.createBackgroundSubtractorMOG2(history=HISTORY, varThreshold=VAR_THRESHOLD,
                                              detectShadows=True)


def carregar_fundos():
    """Le todas as imagens da pasta fundos/. Se estiver vazia, usa um verde solido."""
    arquivos = sorted(glob.glob(os.path.join(PASTA_FUNDOS, "*.*")))
    fundos = [(os.path.basename(a), cv2.imread(a)) for a in arquivos]
    fundos = [(nome, img) for nome, img in fundos if img is not None]  # ignora o que nao e imagem
    if not fundos:
        fundos = [("verde", np.full((ALTURA, ALTURA, 3), (0, 255, 0), np.uint8))]
    return fundos


def preparar(frame):
    """Redimensiona para ALTURA mantendo a proporcao (nao distorce video em pe)."""
    h, w = frame.shape[:2]
    return cv2.resize(frame, (int(w * ALTURA / h), ALTURA))


def aprender_fundo(subtrator, frame):
    """FASE 1: mostra o frame ao MOG2 com learningRate=-1 (ele ajusta o modelo sozinho)."""
    suave = cv2.GaussianBlur(frame, (BLUR, BLUR), 0)
    subtrator.apply(suave, learningRate=-1)


def exposicao_automatica(cap, ligar):
    """So na WEBCAM. Com a pessoa perto (camiseta escura enchendo a imagem), a webcam clareia
    TUDO sozinha: no nosso teste a parede foi de ~130 para ~235 e parte estourou em 255. Isso
    nao se desfaz com um ganho (compensar_exposicao), entao TRAVAMOS a exposicao enquanto o
    fundo aprendido vale. Convencao do OpenCV: 0.75 = automatico, 0.25 = manual."""
    if ligar:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
        return
    alvo = cap.read()[1].mean()              # brilho que o automatico escolheu para a cena vazia
    atual = cap.get(cv2.CAP_PROP_EXPOSURE)   # exposicao informada pela camera (aproximada)
    if not cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25):
        print("Aviso: esta webcam nao deixou travar a exposicao (fica so a compensacao).")
        return
    # O valor informado nao e exato: na nossa webcam dizia -6, mas o brilho do automatico batia
    # com -5 ou -4, conforme a luz. Testamos 5 valores vizinhos (cada +1 dobra o tempo de
    # exposicao) e ficamos com o que deixa o brilho mais parecido com o alvo.
    melhor, menor_dif = atual, 999
    for valor in (atual - 1, atual, atual + 1, atual + 2, atual + 3):
        cap.set(cv2.CAP_PROP_EXPOSURE, valor)
        for _ in range(5):                   # a camera leva alguns frames para mudar
            cap.read()
        dif = abs(cap.read()[1].mean() - alvo)
        if dif < menor_dif:
            melhor, menor_dif = valor, dif
    cap.set(cv2.CAP_PROP_EXPOSURE, melhor)
    print(f"Exposicao da webcam travada em {melhor:g}.")


def compensar_exposicao(subtrator, frame):
    """A camera (celular ou webcam) muda o brilho e as cores sozinha (exposicao e balanco de
    branco automaticos) quando a pessoa entra. Para o MOG2 isso parece que a parede inteira
    mudou. Corrigimos multiplicando cada cor do frame por um ganho que a deixa igual ao fundo."""
    fundo = subtrator.getBackgroundImage()  # a imagem de fundo que o MOG2 aprendeu
    # razao = quanto cada pixel precisa ser multiplicado para ficar igual ao fundo aprendido
    # ([::4, ::4] = usa 1 a cada 4 pixels, so para ficar mais rapido)
    razao = (fundo[::4, ::4].astype(np.float32) + 1) / (frame[::4, ::4].astype(np.float32) + 1)
    # Ideia: se a camera clareou tudo 20%, TODOS os pixels do fundo pedem o MESMO ganho (~0,83),
    # seja parede, movel ou quadro. Ja os pixels da pessoa pedem ganhos variados. Entao o ganho
    # certo e o MAIS COMUM (a moda = o pico do histograma), mesmo se a pessoa ocupar mais da
    # metade da imagem (com a media ou a mediana a pessoa "puxaria" o valor).
    ganho = [1.0, 1.0, 1.0, 0.0]  # B, G, R (o 4o valor e exigido pelo OpenCV e nao e usado)
    for c in range(3):  # um ganho por cor (B, G, R): o balanco de branco muda cada cor de um jeito
        contagem, bordas = np.histogram(razao[:, :, c], bins=75, range=(0.5, 2.0))
        i = np.argmax(contagem)                           # a faixa de ganho com mais pixels
        ganho[c] = float(bordas[i] + bordas[i + 1]) / 2   # o centro dessa faixa
    # multiplica cada cor pelo seu ganho; o OpenCV ja limita o resultado a 255
    return cv2.multiply(frame, tuple(ganho), dtype=cv2.CV_8U)


def segmentar(subtrator, frame):
    """FASE 2: devolve a mascara binaria da pessoa e quantos objetos foram encontrados."""
    # 0) Compensa a exposicao automatica da camera (ver funcao acima)
    if COMPENSAR_EXPOSICAO:
        frame = compensar_exposicao(subtrator, frame)
    # 1) Blur: o ruido do sensor faz pixels "piscarem"; suavizar evita falsos positivos
    suave = cv2.GaussianBlur(frame, (BLUR, BLUR), 0)
    # 2) MOG2 com learningRate=0: o modelo esta CONGELADO, entao uma pessoa parada nao e
    #    incorporada ao fundo. Saida: 0 = fundo, 127 = sombra, 255 = pessoa
    bruta = subtrator.apply(suave, learningRate=0)
    # 3) Limiarizacao: so o que e 255 vira branco; a sombra (127) vira preto
    _, binaria = cv2.threshold(bruta, LIMIAR, 255, cv2.THRESH_BINARY)
    # 4) Morfologia: ABERTURA (erosao -> dilatacao) apaga pontinhos soltos;
    #    FECHAMENTO (dilatacao -> erosao) tapa buracos dentro da silhueta
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KERNEL, KERNEL))
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, k)
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, k)
    # 5) Contornos externos: fica so o que tem area >= AREA_MIN, desenhado PREENCHIDO
    #    (preencher fecha buracos que a morfologia nao fechou). Contar = descritor simples.
    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    grandes = [c for c in contornos if cv2.contourArea(c) >= AREA_MIN]
    mascara = np.zeros_like(binaria)
    cv2.drawContours(mascara, grandes, -1, 255, thickness=cv2.FILLED)
    # 6) Buracos grandes: preencher tapa TODO buraco, inclusive o vao entre o braco e o corpo
    #    quando a mao encosta no quadril (a parede antiga aparecia ali). Buracos = o que ficou
    #    dentro do contorno preenchido mas nao era pessoa; os maiores que BURACO_MAX voltam a 0.
    buracos = cv2.bitwise_and(mascara, cv2.bitwise_not(binaria))
    contornos_b, _ = cv2.findContours(buracos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vaos = [c for c in contornos_b if cv2.contourArea(c) > BURACO_MAX]
    cv2.drawContours(mascara, vaos, -1, 0, thickness=cv2.FILLED)
    return mascara, len(grandes)


def remover_fundo(frame, mascara, fundo):
    """Composicao com borda suave: resultado = alfa*frame + (1-alfa)*fundo."""
    fundo = cv2.resize(fundo, (frame.shape[1], frame.shape[0]))
    # Encolhe a mascara ~2 px: a borda da deteccao sai um pouco para fora da pessoa (por causa
    # do blur), e a parte de fora do borrado mostraria um anel da parede antiga (halo claro)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ENCOLHER_BORDA, ENCOLHER_BORDA))
    mascara = cv2.erode(mascara, k)
    # Borrar a mascara cria valores intermediarios na borda (0..255) -> alfa entre 0 e 1
    alfa = cv2.GaussianBlur(mascara, (SUAVIZAR_BORDA, SUAVIZAR_BORDA), 0).astype(np.float32) / 255
    # blendLinear faz exatamente alfa*frame + (1-alfa)*fundo, pixel a pixel (bem mais rapido
    # que fazer a mesma conta com numpy)
    return cv2.blendLinear(frame, fundo, alfa, 1 - alfa)


def escrever(img, texto, linha):
    """Texto com contorno preto para ficar legivel em qualquer fundo."""
    y = 28 + 28 * linha
    cv2.putText(img, texto, (10, y), FONTE, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, texto, (10, y), FONTE, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def salvar(nome, n, frame, mascara, resultado):
    """Salva os 3 arquivos usados na avaliacao (a mascara e a binaria, sem suavizar)."""
    os.makedirs(PASTA_RESULTADOS, exist_ok=True)
    sufixo = f"{nome}_{n:04d}.png"
    for prefixo, img in (("frame", frame), ("mascara", mascara), ("resultado", resultado)):
        cv2.imwrite(os.path.join(PASTA_RESULTADOS, f"{prefixo}_{sufixo}"), img)
    print("Salvo:", sufixo)


def main():
    args = [a for a in sys.argv[1:] if a != "--gravar"]
    gravar = "--gravar" in sys.argv
    fonte = args[0] if args else "0"
    webcam = fonte.isdigit()
    cap = cv2.VideoCapture(int(fonte) if webcam else fonte)
    if not cap.isOpened():
        sys.exit(f"Erro: nao consegui abrir '{fonte}'.")
    nome = "webcam" if webcam else os.path.splitext(os.path.basename(fonte))[0]
    fps = cap.get(cv2.CAP_PROP_FPS) or 30  # algumas webcams informam 0
    espera = 1 if webcam else max(1, int(1000 / fps))  # video toca perto da velocidade real

    fundos = carregar_fundos()
    i_fundo = 0
    subtrator = criar_subtrator()
    aprendidos = 0   # frames ja usados na fase 1
    n = 0            # numero do frame (vai no nome dos arquivos salvos)
    gravador = None
    # Momento em que a fase 1 comeca. Na webcam esperamos ESPERA_WEBCAM segundos para a pessoa
    # sair da cena; no video nao precisa (ele ja foi gravado com a cena vazia no comeco).
    inicio = time.time() + (ESPERA_WEBCAM if webcam else 0)
    if webcam:
        print(f"Saia da frente da camera: o fundo sera aprendido em {ESPERA_WEBCAM} s.")

    try:  # o "finally" la embaixo devolve a webcam ao automatico e fecha o video gravado,
          # mesmo se o programa for fechado pelo X da janela ou com Ctrl+C
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            n += 1
            frame = preparar(frame)
            falta = inicio - time.time()  # segundos que faltam para comecar a aprender

            if falta > 0:                     # ---------- FASE 0 (so webcam): contagem regressiva
                mascara = np.zeros(frame.shape[:2], np.uint8)
                resultado = frame.copy()
                textos = [f"Saia da cena: {int(falta) + 1}", "o fundo sera aprendido"]
            elif aprendidos < FRAMES_APRENDER:  # ---------- FASE 1: aprender o fundo
                if aprendidos == 0 and webcam and TRAVAR_EXPOSICAO:
                    exposicao_automatica(cap, ligar=False)  # trava antes do 1o frame aprendido
                aprender_fundo(subtrator, frame)
                aprendidos += 1
                mascara = np.zeros(frame.shape[:2], np.uint8)
                resultado = frame.copy()
                textos = [f"Aprendendo fundo {aprendidos}/{FRAMES_APRENDER}", "Fique fora da cena!"]
                if aprendidos == FRAMES_APRENDER:
                    print("Fundo aprendido! Pode entrar na cena.")
            else:                             # ---------- FASE 2: segmentar
                mascara, n_objetos = segmentar(subtrator, frame)
                resultado = remover_fundo(frame, mascara, fundos[i_fundo][1])
                textos = [f"Fundo: {fundos[i_fundo][0]}", f"Objetos: {n_objetos}"]
                if not webcam and SALVAR_A_CADA and n % SALVAR_A_CADA == 0:
                    salvar(nome, n, frame, mascara, resultado)

            # Painel: original | mascara | resultado (texto so na copia mostrada)
            tela_res = resultado.copy()
            for i, t in enumerate(textos):
                escrever(tela_res, t, i)
            painel = np.hstack([frame, cv2.cvtColor(mascara, cv2.COLOR_GRAY2BGR), tela_res])
            cv2.imshow(JANELA, cv2.resize(painel, None, fx=ESCALA_TELA, fy=ESCALA_TELA))
            if gravar:
                if gravador is None:  # cria na 1a vez, quando ja sabemos o tamanho do painel
                    os.makedirs(PASTA_RESULTADOS, exist_ok=True)
                    gravador = cv2.VideoWriter(os.path.join(PASTA_RESULTADOS, f"demo_{nome}.mp4"),
                                               cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                               (painel.shape[1], painel.shape[0]))
                gravador.write(painel)

            tecla = cv2.waitKey(espera) & 0xFF
            if tecla == ord("q") or cv2.getWindowProperty(JANELA, cv2.WND_PROP_VISIBLE) < 1:
                break  # Q ou janela fechada no X
            elif tecla == ord("d"):  # proximo fundo (igual ao ClearCut)
                i_fundo = (i_fundo + 1) % len(fundos)
            elif tecla == ord("a"):  # fundo anterior
                i_fundo = (i_fundo - 1) % len(fundos)
            elif tecla == ord("b"):  # recomeca a fase 1 com um MOG2 vazio (com contagem na webcam)
                subtrator, aprendidos = criar_subtrator(), 0
                inicio = time.time() + (ESPERA_WEBCAM if webcam else 0)
                if webcam and TRAVAR_EXPOSICAO:
                    exposicao_automatica(cap, ligar=True)  # reajusta durante a contagem
                print("Reaprendendo o fundo: saia da cena.")
            elif tecla == ord("s") and aprendidos >= FRAMES_APRENDER:
                salvar(nome, n, frame, mascara, resultado)
    finally:
        if webcam and TRAVAR_EXPOSICAO:
            exposicao_automatica(cap, ligar=True)  # devolve a webcam como estava (outros programas)
        cap.release()
        if gravador is not None:
            gravador.release()  # sem isso o .mp4 fica corrompido
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
