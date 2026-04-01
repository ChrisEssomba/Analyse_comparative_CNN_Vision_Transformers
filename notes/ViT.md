Dataset
Le jeu de donnée est une extraction de tiny-imagenet-200 (jeu de données connu par la communauté)
Une extraction de 10 classes. 5 dont nous avons jugé adapté pour la texture et 5 pour la dépendence à long terme (images à l'appui). Datasets riches qui posèdes des images dans des conditions très varié (objets partiellements cachés, image dégradé, etc). Le jeu de donnée constitué est de 3500 image d'entrainement 750 images de validation et 750 image de test.


Architecture Globale

1 Extraction et Projection des Patches (Patch Embedding)
L'image d'entrée (64×64×3 canaux) est partitionnée en patches de taille 8×8 pixels. Patch embedding convolutif hybride : Convolution dans la formation des patches + agrégation dynamique des features voisines pour capturer la localité spatiale
Chaque patch, est projeté vers un espace d'embedding de 192 dimensions .

2 Token de Classification et Embeddings Positionnels
Token CLS : Un vecteur de 192 dimensions est préfixé à la séquence de patches, formant une séquence de 65 embeddings [CLS, patch1, ..., patch8].
Ce token CLS agit comme un récepteur global d'information via l'auto-attention.
Embeddings positionnels : Des embeddings positionnels de taille [1, 65, 192] sont ajoutés élément par élément,
permettant au modèle de distinguer l'ordre spatial des patches.

3 Blocs Transformer (Encoder)
Quatre couches TransformerEncoderLayer identiques, chacune comprenant :

- Attention Multi-Tête : 8 têtes d'attention avec 24 dimensions par tête (192/8)

- MLP Feed-Forward : (192 → 768 → 192) avec activation GELU

- Normalisation Pre-LayerNorm : Normalisation précédant chaque sous-bloc

- Dropout 25% : Régularisation agressive adaptée aux petits datasets

4 Tête de Classification
Extraction : Seul le token CLS raffiné
Projection finale : MLP Convolutif produisant les logits de classification.


