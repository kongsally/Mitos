import os
import argparse
import numpy as np
from PIL import Image, ImageDraw
from Bio import SeqIO

# Various Tie-ups to play with 
straightTieUp = np.array([
    [0, 0, 0, 1], 
    [0, 0, 1, 0], 
    [0, 1, 0, 0], 
    [1, 0, 0, 0]
    ])
basketTieUp = np.array([
    [0, 0, 1, 1], 
    [0, 0, 1, 1],
    [1, 1, 0, 0], 
    [1, 1, 0, 0]
])
crowTieUp = np.array([
    [0, 0, 1, 1], 
    [1, 0, 0, 1], 
    [0, 1, 1, 0], 
    [1, 1, 0, 0]
])
twillTieUp = np.array([
    [1, 1, 0, 0], 
    [0, 1, 1, 0], 
    [0, 0, 1, 1], 
    [1, 0, 0, 1]
])

basketWeave = np.array(
    [[0, 0, 1, 1], 
     [0, 0, 1, 1], 
     [1, 1, 0, 0], 
     [1, 1, 0, 0],
    ]
)
twillWeave = np.array(
    [
        [1, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 1, 1],
        [1, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 1, 1],
        [1, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 1, 1],
        [1, 0, 0, 1],
    ]
)
twillDiamond = np.vstack((
    twillWeave,
    np.flip(twillWeave, axis=0)
))

# D Loop Sequences from NCBI
DLoop_L0 = "CTGCCAGCCACCATGAATATTGTACAGTACCATAAATACTTGACTACCTGCAGTACATAAAAACTCAACCCACATCAAAACCCTGCCCCCATGCTTACAAGCAAGTACAGCAATCAACCTTCAACTGTCACACATCAACTGCAACTCCAAAGCCACCTCTCACCCACTAGGATACCAACAAACTTACCCACCCTTAACAGTACATAGCACATAAAGTCATTTACCGTACATAGCACATTACAGTCAAATCCCTTCTCGTCCCCATGGATGACCCCCCTCAGATAGGGGTCCCTTGACCACCATCCTCCGTGAAATCAATATCCCGCACAAGAGTGCTACTCTCCTCGCTCCGGGCCCATAACACTTGGGGGTAGCTAAAGTGAACTGTATCCGACATCTGGTTCCTACTTCAGGGTCATAAAGCCTAAATAGCCCACACGTTCCCCTTAAATAAGACATCACGATG"
DLoop_K1 = "TTCTTTCATGGGGAAGCAGATTTGGGTACCACCCAAGTATTGACTCACCCATCAACAACCGCTATGTATCTCGTACATTACTGCCAGCCACCATGAATATTGTACGGTACCATAAATACTTGACCACCTGTAGTACATAAAAACCCAATCCACATCAAAACCCCCTCCCCATGCTTACAAGCAAGTACAGCAATCAACCCCCAACTATCACACATCAACTGCAACTCCAAAGCCACCCCTCACCCACTAGGATACCAACAAACCTACCCACCCTTAACAGTACATAGCACATAAAGCCATTTACCGTACATAGCACATTACAGTCAAATCCCTTCTCGTCCCCATGGATGACCCCCCTCAGATAGGGGTCCCTTGACCACCATCCTCCGTGAAATCAATATCCCGCACAAGAGTGCTACTCTCCTCGCTCCGGGCCCATAACACTTGGGGGTAGCTAAAGTGAACTGTATCCGACATCTGGTTCCTACTTCAGGGCCATAAAGCCTAAATAGCCCACACGTTCCCCTTAAATAAGACATCACGATG"
DLoop_M8 = "TTCTTTCATGGGGAAGCAGATTTGGGTACCACCCAAGTATTGACTCACCCATCAACAACCGCTATGTATTTCGTACATTACTGCCAGCCACCATGAATATTGTACGGTACCATAAATACTTGACCACCTGTAGTACATAAAAACCCAATCCACATCAAAATCCCCTCCCCATGCTTACAAGCAAGTACAGCAATCAACCTTCAACTATCACACATCAACTGCAACTCCAAAGCCACCCCTCACCCACTAGGATACCAACAAACCTACCCACCCTCAACAGTACATAGTACATAAAACCATTTACCGTACATAGCACATTACAGTCAAATCCCTTCTCGTCCCCATGGATGACCCCCCTCAGATAGGGGTCCCTTGACCACCATCCTCCGTGAAATCAATATCCCGCACAAGAGTGCTACTCTCCTCGCTCCGGGCCCATAACACTTGGGGGTAGCTAAAGTGAACTGTATCCGACATCTGGTTCCTACTTCAGGGTCATAAAGCCTAAATAGCCCACACGTTCCCCTTAAATAAGACATCACGATG"

def readFasta(fasta_file):
    """
    Returns the DNA Sequence from the given fasta file 

    Args:
        fasta_file
    """
    if not os.path.exists(fasta_file):
        raise FileNotFoundError(f"{args.fasta_file} file was not found")
    
    with open(fasta_file, 'r') as file:
        return "".join([str(record.seq) for record in SeqIO.parse(file, 'fasta')])

def dnaSeqToThreadingMatrix(dna_seq):
    """
    Returns a threading matrix based on the given DNA Sequence where
    A, T, G, C will be mapped the 1, 2, 3, 4th shafts

    Args:
        dna_seq (str): DNA Sequence to be mapped
    
    Returns:
        np.array (4, len(dna_seq))

    """
    # Mapping for the four nucleotides to the 4 shafts of the loom, A:1, T:2, G:3, C:4
    geneToShaft = {
        "A": [1, 0, 0, 0], 
        "T": [0, 1, 0, 0], 
        "G": [0, 0, 1, 0], 
        "C": [0, 0, 0, 1]
    }
    return np.array([geneToShaft[x] for x in dna_seq]).T

def dnaSeqCompTreadlingMatrix(dna_seq):
    """
    Returns a treadling matrix based on the given DNA Sequence where
    A, T, G, C will be mapped the 1, 2, 3, 4th shafts

    Args:
        dna_seq (str): DNA Sequence to be mapped
    
    Returns:
        np.array (4, len(dna_seq))

    """
    # Treadling Patterns
    compGeneDict = {
        "A": [0, 1, 0, 0], 
        "T": [1, 0, 0, 0], 
        "G": [0, 0, 0, 1], 
        "C": [0, 0, 1, 0]
        }
    geneCompWeave = np.array([compGeneDict[x] for x in dna_seq])
    return geneCompWeave


def drawFullDraft(fullDraftMatrix, cellSize=20):

    # Compute the size of the resulting image
    width = fullDraftMatrix.shape[1] * cellSize
    height = fullDraftMatrix.shape[0] * cellSize

    # Create a new blank image with the computed size
    image = Image.new("RGB", (width, height), color="white")

    COLOR_THREADING = "blue"
    COLOR_TIEUP = "green"
    COLOR_TREADLING = "black"
    COLOR_DRAWDOWN =  "orange"

    # Draw the cells based on the binary matrix
    draw = ImageDraw.Draw(image)
    for row in range(fullDraftMatrix.shape[0]):
        for col in range(fullDraftMatrix.shape[1]):
            x = col * cellSize
            y = row * cellSize

            if fullDraftMatrix[row, col]:
                # Last four columns for tie-up and treadling
                if col >= fullDraftMatrix.shape[1] - 4:
                    if row < 4:
                        fill_color = COLOR_TIEUP
                    else:
                        fill_color = COLOR_TREADLING

                # First four rows for the threading pattern
                elif row < 4:   
                    fill_color = COLOR_THREADING 
                else: 
                    fill_color = COLOR_DRAWDOWN
                draw.rectangle([(x, y), (x + cellSize, y + cellSize)], fill=fill_color)
                #draw.text((x,y), "X", fill=fill_color)
    return image

if __name__ == "__main__":

    # Parse input argument
    parser = argparse.ArgumentParser()
    parser.add_argument("fasta_file", help="path to input FASTA file")
    args = parser.parse_args()
    dna = readFasta(args.fasta_file)

    # Create the Threading Matrix based on the DNA Sequence
    threadingMatrix = dnaSeqToThreadingMatrix(dna)
    tieUpMatrix = straightTieUp
    treadlingMatrix = dnaSeqCompTreadlingMatrix(dna)
    drawDownMatrix = treadlingMatrix @ tieUpMatrix.T @ threadingMatrix

    # The Full Draft Matrix consists of the threading, tie-up, drawdown, and treadling matrices
    fullDraftMatrix = np.vstack(
        (
            np.hstack((threadingMatrix, tieUpMatrix)),
            np.hstack((drawDownMatrix, treadlingMatrix)),
        )
    )

    # Draw out the full draft
    weavingDraftImg = drawFullDraft(fullDraftMatrix, cellSize=10)
    weavingDraftImg.show()
