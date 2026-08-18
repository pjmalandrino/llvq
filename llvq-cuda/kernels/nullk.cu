// tv_nullk: le PLANCHER — la même passe, sans un octet de poids.
//
// Bras `nullk` de `proofs/preregistration-p4-2026-08-14.md` §2.5 : *même
// grille, même k, sortie écrite, aucune lecture de poids*.
//
// ## Ce qu'il mesure, et pourquoi c'est le seul bras qui le mesure
//
// L'attribution du gisement CUDA découpe les 2,04 ms/token en **latence et
// occupation 39 %, flux 33 %, décodage 19 %**. Les quatre tentatives de
// descendre sous `Planes14` — E3, `Golay70` v2, `e1c14`, E1v — attaquaient
// toutes les 33 % en gonflant les 19 %. **Personne n'a jamais attaqué les
// 39 %**, et personne ne les a jamais mesurés directement : ils sont un reste,
// obtenu par soustraction dans une attribution de 2026-08-05.
//
// Ce noyau est ce reste, rendu observable. Il garde TOUT ce qu'un bras LLVQ
// fait autour du décodage :
//
//   * la même grille — un warp par ligne, 8 lignes par bloc de 256 ;
//   * le même tuilage et les deux mêmes `__syncthreads` ;
//   * le même staging de l'activation en mémoire partagée ;
//   * la même réduction `warp_sum` et le même épilogue de queue ;
//   * la même écriture de `y`.
//
// et il retire une seule chose : **la lecture et le décodage du bloc**.
//
//     t(planes14) − t(nullk)  =  trafic de poids + décodage
//     t(nullk)                =  tout le reste, c'est-à-dire les 39 %
//
// ## Le piège, et ce qui le désamorce
//
// 🚨 Un noyau qui ne lit aucun poids est un noyau que le compilateur peut
// vider. S'il vidait la boucle de tuiles, il supprimerait le staging — donc
// précisément la moitié de ce que ce bras existe pour mesurer — et rendrait un
// plancher flatteur que rien dans la sortie ne signalerait.
//
// Il accumule donc l'ACTIVATION STAGÉE elle-même : `xs[j]` est lu depuis la
// mémoire partagée, le résultat part dans `y`, et la chaîne
// global → partagé → registre → global est complète. Le staging est
// load-bearing par construction, pas par espoir.
//
// ⚠️ Ce que ce bras ne peut PAS être : vérifié contre la référence f64. Il ne
// calcule pas le produit du modèle et n'a aucun étalon — comme `sol` dans
// `bin/rankbench`, c'est une ancre, et ce qu'on exige de lui est d'être
// OBSERVABLE, pas juste. Le banc le traite ainsi explicitement.
//
// ## Ce qu'il ne mesure pas
//
// Ni la latence de lancement seule (108 lancements en moins valent 0,392 ms
// mesurés ailleurs), ni l'occupation seule. Il rend leur somme plus le staging
// plus la réduction — un plafond sur ce qu'un décodage parfaitement gratuit
// laisserait, et c'est exactement la quantité qui manque au dossier.
//
// Même contrat d'assemblage que `planes.cu` : NVRTC n'a pas de système de
// fichiers, l'hôte concatène, et les gardes ci-dessous ne résolvent depuis le
// disque que sous une vérification hôte (`bin/cuhcheck`).

#ifndef TILE_COLS
#include "matvec.cu"
#endif

extern "C" __global__ void tv_nullk(const float* __restrict__ rscale,
                                    const float* __restrict__ tail,
                                    const float* __restrict__ x,
                                    float* __restrict__ y,
                                    u32 nblocks,
                                    u32 tail_w)
{
    extern __shared__ float xs[];
    u32 lane = threadIdx.x & 31u;
    u32 row  = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    float acc = 0.0f;

    u32 ntiles = (nblocks + TILE_BLOCKS - 1u) / TILE_BLOCKS;
    for (u32 t = 0; t < ntiles; ++t) {
        u32 jlo = t * TILE_BLOCKS;
        u32 jhi = jlo + TILE_BLOCKS < nblocks ? jlo + TILE_BLOCKS : nblocks;
        u32 n   = (jhi - jlo) * LLVQ_DIM;
        __syncthreads();
        for (u32 i = threadIdx.x; i < n; i += blockDim.x) xs[i] = x[jlo * LLVQ_DIM + i];
        __syncthreads();

        // La boucle de `tv_planes`, au décodage près. Chaque lane touche les 24
        // créneaux de ses blocs — même nombre d'itérations, même accès à la
        // partagée — et n'ouvre aucun mot de poids.
        for (u32 j = jlo + lane; j < jhi; j += 32u) {
            const float* xb = xs + (j - jlo) * LLVQ_DIM;
#pragma unroll
            for (u32 s = 0; s < LLVQ_DIM; ++s) acc = __fmaf_rn(1.0f, xb[s], acc);
        }
    }

    acc = warp_sum(acc);
    if (lane == 0) {
        float tv = 0.0f;
        u32 tc0 = nblocks * LLVQ_DIM;
        for (u32 i = 0; i < tail_w; ++i) tv += tail[row * tail_w + i] * x[tc0 + i];
        y[row] = acc * rscale[row] + tv;
    }
}
