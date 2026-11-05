"""
Cria as mascaras GABARITO (a resposta certa) pintando a pessoa com o mouse.

Para cada resultados/frame_*.png que ainda nao tem gabarito, abre o frame e voce pinta
por cima da pessoa. O gabarito e salvo em gabarito/ com o MESMO nome final do frame
(frame_video1_0120.png -> gabarito_video1_0120.png), branco = pessoa, preto = fundo.

Uso: python codigo/criar_gabarito.py                       -> todos os frames sem gabarito
     python codigo/criar_gabarito.py claro_0520 escuro_0400 -> so os frames com esses textos no nome
                                                               (se ja tiverem gabarito, ele abre o
                                                               salvo para CORRIGIR, sem comecar do zero)
Mouse: esquerdo = pinta | direito = apaga
Teclas: + / - = tamanho do pincel | S = salvar e ir para o proximo | N = pular | Q = sair
"""
import glob
import os
import sys

import cv2
import numpy as np

PINCEL_INICIAL = 6    # raio do pincel em pixels do frame (mude com + e -). Pincel grande no
                      # meio do corpo e pequeno nas bordas: o grande passa da pessoa e pinta parede.
ZOOM = 1.5            # a imagem aparece ampliada (so na tela; o gabarito e salvo no tamanho do
                      # frame). Em 1:1 o frame tem ~270 px de largura e fica dificil acertar a
                      # borda. Se a janela nao couber na tela, diminua (1 = sem ampliar).
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASTA_RESULTADOS = os.path.join(RAIZ, "resultados")
PASTA_GABARITO = os.path.join(RAIZ, "gabarito")
JANELA = "Gabarito: pinte a pessoa (S salva, N pula, Q sai, +/- pincel)"

# Variaveis compartilhadas entre o laco principal e a funcao do mouse
estado = {"mascara": None, "pincel": PINCEL_INICIAL, "cor": None, "pos": (0, 0)}


def mouse(evento, x, y, flags, param):
    """Chamada pelo OpenCV a cada evento do mouse: pinta 255 ou apaga 0 na mascara."""
    x, y = int(x / ZOOM), int(y / ZOOM)  # a tela esta ampliada: volta para o pixel do frame
    if evento == cv2.EVENT_LBUTTONDOWN:
        estado["cor"] = 255
    elif evento == cv2.EVENT_RBUTTONDOWN:
        estado["cor"] = 0
    elif evento in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
        estado["cor"] = None
    if estado["cor"] is not None:
        # linha do ponto anterior ate o atual: nao deixa "falhas" se o mouse andar rapido
        cv2.line(estado["mascara"], estado["pos"], (x, y), estado["cor"], 2 * estado["pincel"])
    estado["pos"] = (x, y)


def mostrar(frame):
    """Frame ampliado com a area pintada em vermelho semitransparente + circulo do pincel."""
    tela = frame.copy()
    pintado = estado["mascara"] == 255
    tela[pintado] = (0.5 * tela[pintado] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
    tela = cv2.resize(tela, None, fx=ZOOM, fy=ZOOM)
    x, y = estado["pos"]
    cv2.circle(tela, (int(x * ZOOM), int(y * ZOOM)), int(estado["pincel"] * ZOOM),
               (255, 255, 255), 1)
    return tela


def main():
    os.makedirs(PASTA_GABARITO, exist_ok=True)
    frames = sorted(glob.glob(os.path.join(PASTA_RESULTADOS, "frame_*.png")))
    destino = lambda f: os.path.join(PASTA_GABARITO,
                                     os.path.basename(f).replace("frame_", "gabarito_", 1))
    filtros = sys.argv[1:]  # opcional: escolher os frames (ex.: claro_0520), sem pular um por um
    if filtros:   # frames escolhidos: abre mesmo os que ja tem gabarito (para corrigir)
        pendentes = [f for f in frames if any(t in os.path.basename(f) for t in filtros)]
    else:         # sem escolher: so os que ainda nao tem gabarito
        pendentes = [f for f in frames if not os.path.exists(destino(f))]
    print(f"{len(frames)} frames em resultados/, {len(pendentes)} para pintar.")
    if not pendentes:
        return
    cv2.namedWindow(JANELA)
    cv2.setMouseCallback(JANELA, mouse)

    for caminho in pendentes:
        frame = cv2.imread(caminho)
        if os.path.exists(destino(caminho)):  # ja tem gabarito: comeca dele, para corrigir
            estado["mascara"] = cv2.imread(destino(caminho), cv2.IMREAD_GRAYSCALE)
            print("Corrigindo:", os.path.basename(caminho), "(N = deixar como estava)")
        else:
            estado["mascara"] = np.zeros(frame.shape[:2], np.uint8)  # tudo preto (fundo)
            print("Pintando:", os.path.basename(caminho))
        while True:
            cv2.imshow(JANELA, mostrar(frame))
            tecla = cv2.waitKey(20) & 0xFF
            if tecla in (ord("+"), ord("=")):
                estado["pincel"] = min(80, estado["pincel"] + 2)
            elif tecla == ord("-"):
                estado["pincel"] = max(1, estado["pincel"] - 2)
            elif tecla == ord("s"):
                cv2.imwrite(destino(caminho), estado["mascara"])
                print("  salvo:", os.path.basename(destino(caminho)))
                break
            elif tecla == ord("n"):
                break
            elif tecla == ord("q"):
                cv2.destroyAllWindows()
                return
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
