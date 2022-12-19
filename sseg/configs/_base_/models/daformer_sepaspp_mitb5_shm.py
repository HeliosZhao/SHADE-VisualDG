# DAFormer (with context-aware feature fusion) in Tab. 7

_base_ = ['daformer_sepaspp_mitb5.py']

norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    backbone=dict(type='mit_b5', 
    style='pytorch',
    shm_cfg=dict(
        concentration_coeff=0.0156,
        base_style_num=64,
        layer=1,
    )),
    )
