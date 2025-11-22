# PainterI2V avec Encodage VAE Tiled

## 📦 Fichiers fournis

### 1. `nodes_painter_i2v_tiled.py`
Contient **uniquement** la nouvelle version avec encodage tiled :
- **PainterI2VTiled** : Version améliorée avec support de l'encodage VAE par tuiles

### 2. `nodes_complete.py` 
Contient **les deux versions** :
- **PainterI2V** : Version originale (encodage VAE standard)
- **PainterI2VTiled** : Version avec encodage tiled

## 🎯 Quelle version choisir ?

### Utilisez `PainterI2VTiled` si :
- ✅ Vous générez des vidéos longues (> 81 frames)
- ✅ Vous utilisez des résolutions élevées (> 832x480)
- ✅ Vous rencontrez des erreurs "Out of Memory" (OOM)
- ✅ Vous voulez optimiser l'utilisation de la VRAM

### Restez sur `PainterI2V` si :
- ✅ Vous générez des vidéos courtes (≤ 81 frames)
- ✅ Résolutions standard (832x480 ou moins)
- ✅ Vous avez suffisamment de VRAM (12+ GB)
- ✅ Vous préférez la simplicité (moins de paramètres)

## 🔧 Installation

1. Copiez le fichier choisi dans votre dossier `ComfyUI/custom_nodes/`
2. Renommez-le en `nodes.py` (ou le nom de votre choix)
3. Redémarrez ComfyUI

## 📊 Nouveaux paramètres Tiled

Le node `PainterI2VTiled` ajoute 4 paramètres pour l'encodage par tuiles :

### **tile_size** (défaut: 512)
- Taille des tuiles spatiales en pixels
- Plus petit = moins de VRAM, plus lent
- Recommandé : 512 pour la plupart des cas

### **overlap** (défaut: 64)
- Chevauchement entre tuiles spatiales en pixels
- Évite les artefacts visibles aux jonctions
- Recommandé : 64-128 pixels

### **temporal_size** (défaut: 64)
- Nombre de frames encodées simultanément
- Plus petit = moins de VRAM pour longues vidéos
- Recommandé : 64 pour vidéos < 200 frames, 32 pour plus longues

### **temporal_overlap** (défaut: 8)
- Chevauchement entre chunks temporels
- Évite les "sauts" entre segments
- Recommandé : 8-16 frames

## 🎨 Fonctionnalités conservées

Les deux versions incluent :
- ✨ **Motion Amplitude** : Correction du slow-motion des LoRAs 4-step
- 🎯 **Reference Latents** : Amélioration de la cohérence de la première frame
- 🖼️ **CLIP Vision** : Support des embeddings visuels
- 🔄 **Batch Processing** : Génération multiple

## 💡 Exemples de configuration

### Configuration rapide (vidéos courtes)
```
length: 81
tile_size: 512
overlap: 64
temporal_size: 64
temporal_overlap: 8
```

### Configuration économe en VRAM (vidéos longues)
```
length: 200+
tile_size: 384
overlap: 64
temporal_size: 32
temporal_overlap: 8
```

### Configuration haute qualité (VRAM abondante)
```
length: 81-161
tile_size: 640
overlap: 128
temporal_size: 64
temporal_overlap: 16
```

## ⚠️ Notes importantes

1. **L'encodage tiled est légèrement plus lent** que l'encodage standard, mais évite les crashs mémoire
2. **Ne modifiez pas motion_amplitude** si vous n'utilisez pas de LoRA 4-step (gardez 1.0)
3. **Overlap trop petit** peut créer des artefacts de grille visibles
4. **Temporal_size trop petit** peut créer des discontinuités temporelles

## 🐛 Dépannage

### OOM même avec tiled encoding ?
- Réduisez `tile_size` à 384 ou 256
- Réduisez `temporal_size` à 32 ou 16
- Vérifiez que vous n'avez pas d'autres processus gourmands en VRAM

### Artefacts visibles ?
- Augmentez `overlap` à 96 ou 128
- Augmentez `temporal_overlap` à 12 ou 16

### Génération trop lente ?
- Augmentez `tile_size` si vous avez la VRAM
- Augmentez `temporal_size`

## 📝 Changelog

### Version Tiled (nouvelle)
- ➕ Ajout encodage VAE tiled (spatial + temporel)
- ➕ Support vidéos longues sans OOM
- ➕ 4 nouveaux paramètres configurables
- ✅ Conservation de toutes les fonctionnalités de PainterI2V

### Version originale
- ✅ Fix slow-motion pour LoRAs 4-step
- ✅ Motion amplitude avec protection luminosité
- ✅ Reference latents pour cohérence

## 🤝 Crédits

- **PainterI2V original** : Votre custom node
- **Encodage Tiled** : Basé sur ComfyUI-WanImageToVideoTiled
- **Fusion** : Combinaison des deux approches

## 📧 Support

Si vous rencontrez des problèmes :
1. Vérifiez que votre version de ComfyUI est à jour
2. Testez d'abord avec les paramètres par défaut
3. Ajustez progressivement selon vos besoins
