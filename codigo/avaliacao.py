"""
Avaliacao: compara a mascara do PROGRAMA com o GABARITO feito a mao.

Para cada resultados/mascara_<video>_<frame>.png que tenha gabarito/gabarito_<video>_<frame>.png:
  IoU = (pixels de pessoa em comum) / (pixels de pessoa no programa OU no gabarito)
  ACERTO se IoU >= IOU_MIN.
Imprime a tabela e a porcentagem de acertos, e salva resultados/tabela_acertos.csv.

Uso: python codigo/avaliacao.py
"""
import csv
import glob
import os

import cv2
import numpy as np

IOU_MIN = 0.8      # criterio de acerto. Maior = criterio mais exigente (menos acertos).
MIN_FRAMES = 10    # o trabalho exige avaliar pelo menos 10 frames
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASTA_RESULTADOS = os.path.join(RAIZ, "resultados")
PASTA_GABARITO = os.path.join(RAIZ, "gabarito")


def calcular_iou(mascara, gabarito):
    """IoU entre duas mascaras binarias (branco = pessoa)."""
    prog = mascara > 127   # True onde o programa disse "pessoa"
    real = gabarito > 127  # True onde o gabarito diz "pessoa"
    intersecao = np.logical_and(prog, real).sum()
    uniao = np.logical_or(prog, real).sum()
    return 1.0 if uniao == 0 else intersecao / uniao  # os dois vazios = concordam


def main():
    linhas = []
    for arq in sorted(glob.glob(os.path.join(PASTA_RESULTADOS, "mascara_*.png"))):
        chave = os.path.basename(arq)[len("mascara_"):-len(".png")]  # ex.: video1_0120
        arq_gab = os.path.join(PASTA_GABARITO, f"gabarito_{chave}.png")
        if not os.path.exists(arq_gab):
            continue  # frame sem gabarito: nao entra na avaliacao
        mascara = cv2.imread(arq, cv2.IMREAD_GRAYSCALE)
        gabarito = cv2.imread(arq_gab, cv2.IMREAD_GRAYSCALE)
        if gabarito.shape != mascara.shape:  # gabarito feito em outro programa e redimensionado
            gabarito = cv2.resize(gabarito, (mascara.shape[1], mascara.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
        video, frame = chave.rsplit("_", 1)  # o numero do frame vem depois do ultimo "_"
        iou = calcular_iou(mascara, gabarito)
        linhas.append((video, frame, iou, iou >= IOU_MIN))

    if not linhas:
        print("Nenhuma mascara com gabarito. Rode antes o criar_gabarito.py.")
        return
    if len(linhas) < MIN_FRAMES:
        print(f"ATENCAO: so {len(linhas)} frames avaliados (o minimo pedido e {MIN_FRAMES}).")

    print(f"\n{'Video':<20}{'Frame':>6}{'IoU':>8}   Resultado (IoU >= {IOU_MIN})")
    for video, frame, iou, acerto in linhas:
        print(f"{video:<20}{frame:>6}{iou:>8.3f}   {'ACERTO' if acerto else 'ERRO'}")
    print("-" * 50)
    for video in sorted({l[0] for l in linhas}):  # porcentagem separada por video
        do_video = [l for l in linhas if l[0] == video]
        acertos = sum(l[3] for l in do_video)
        print(f"{video}: {acertos}/{len(do_video)} acertos = {100 * acertos / len(do_video):.1f}%")
    acertos = sum(l[3] for l in linhas)
    taxa = 100 * acertos / len(linhas)
    media = np.mean([l[2] for l in linhas])
    print(f"TOTAL: {acertos}/{len(linhas)} acertos = {taxa:.1f}%   (IoU medio = {media:.3f})")

    os.makedirs(PASTA_RESULTADOS, exist_ok=True)
    caminho = os.path.join(PASTA_RESULTADOS, "tabela_acertos.csv")
    # ";" e virgula decimal: abre direto no Excel em portugues; utf-8-sig mantem acentos
    with open(caminho, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["video", "frame", "iou", "resultado"])
        for video, frame, iou, acerto in linhas:
            w.writerow([video, frame, f"{iou:.3f}".replace(".", ","), "ACERTO" if acerto else "ERRO"])
        w.writerow(["TOTAL", len(linhas), f"{media:.3f}".replace(".", ","),
                    f"{taxa:.1f}% de acertos".replace(".", ",")])
    print("Tabela salva em", caminho)


if __name__ == "__main__":
    main()
