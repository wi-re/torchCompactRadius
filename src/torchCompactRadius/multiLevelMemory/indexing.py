import torch

def spread2(w : torch.Tensor):
    w &= 0x00000000001fffff
    w = (w | w << 32) & 0x001f00000000ffff
    w = (w | w << 16) & 0x001f0000ff0000ff
    w = (w | w <<  8) & 0x010f00f00f00f00f
    w = (w | w <<  4) & 0x10c30c30c30c30c3
    w = (w | w <<  2) & 0x1249249249249249
    return w
def compact2(w : torch.Tensor):
    w &= 0x1249249249249249
    w = (w ^ (w >> 2))  & 0x30c30c30c30c30c3
    w = (w ^ (w >> 4))  & 0xf00f00f00f00f00f
    w = (w ^ (w >> 8))  & 0x00ff0000ff0000ff
    w = (w ^ (w >> 16)) & 0x00ff00000000ffff
    w = (w ^ (w >> 32)) & 0x00000000001fffff
    return w
def Z_encode3D(x : torch.Tensor, y : torch.Tensor, z : torch.Tensor):
    return ((spread2(x.to(torch.int64))) | (spread2(y.to(torch.int64)) << 1) | (spread2(z.to(torch.int64)) << 2))
def Z_decode3D(Z_code):
    x = compact2(Z_code)
    y = compact2(Z_code >> 1)
    z = compact2(Z_code >> 2)
    return torch.stack([x, y, z], dim = 1)

def spread1(x: torch.Tensor):
    x &= 0x00000000ffffffff
    x = (x | (x << 16)) & 0x0000FFFF0000FFFF
    x = (x | (x << 8)) & 0x00FF00FF00FF00FF
    x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0F
    x = (x | (x << 2)) & 0x3333333333333333
    x = (x | (x << 1)) & 0x5555555555555555
    return x
def compact1(x: torch.Tensor):
    x = x & 0x5555555555555555
    x = (x | (x >> 1))  & 0x3333333333333333
    x = (x | (x >> 2))  & 0x0F0F0F0F0F0F0F0F
    x = (x | (x >> 4))  & 0x00FF00FF00FF00FF
    x = (x | (x >> 8))  & 0x0000FFFF0000FFFF
    x = (x | (x >> 16)) & 0x00000000FFFFFFFF
    return x
def Z_encode2D(x : torch.Tensor, y : torch.Tensor):
    return ((spread1(x.to(torch.int64))) | (spread1(y.to(torch.int64)) << 1))
def Z_decode2D(Z_code):
    x = compact1(Z_code)
    y = compact1(Z_code >> 1)
    return torch.stack([x, y], dim = 1)


def mortonEncode(i: torch.Tensor):
    if i.shape[1] == 1:
        return i
    elif i.shape[1] == 2:
        return Z_encode2D(i[:,0], i[:,1])
    elif i.shape[1] == 3:
        return Z_encode3D(i[:,0], i[:,1], i[:,2])
    else:
        raise ValueError('Only 1, 2, or 3 dimensions are supported')
    
def mortonDecode(i: torch.Tensor, dim):
    if dim == 1:
        return i
    elif dim == 2:
        return Z_decode2D(i)
    elif dim == 3:
        return Z_decode3D(i)
    else:
        raise ValueError('Only 1, 2, or 3 dimensions are supported')
    
def getMortonCodes(positions, hCell, domainDescription, levels):
    ci = ((positions - domainDescription.min) / hCell).int()
    morton = mortonEncode(ci)

    codes = []
    for level in range(levels):
        codes.append((morton >> level * 2).int())
    return codes


def linearToGrid(linearIndex, gridResolution):
    if len(gridResolution) == 1:
        return linearIndex
    elif len(gridResolution) == 2:
        xi = linearIndex % gridResolution[0]
        yi = linearIndex // gridResolution[0]
        return torch.stack([xi, yi], dim = 1)
    elif len(gridResolution) == 3:
        xi = linearIndex % gridResolution[0]
        yi = (linearIndex // gridResolution[0]) % gridResolution[1]
        zi = linearIndex // (gridResolution[0] * gridResolution[1])
        return torch.stack([xi, yi, zi], dim = 1)
    else:
        raise ValueError('Only 1, 2, or 3 dimensions are supported')

def gridToLinear(gridIndex, gridResolution):
    if len(gridResolution) == 1:
        return gridIndex[:,0]
    elif len(gridResolution) == 2:
        return gridIndex[:,0] + gridIndex[:,1] * gridResolution[0]
    elif len(gridResolution) == 3:
        return gridIndex[:,0] + gridIndex[:,1] * gridResolution[0] + gridIndex[:,2] * gridResolution[0] * gridResolution[1]
    else:
        raise ValueError('Only 1, 2, or 3 dimensions are supported')

def linearToMorton(linearIndex, gridResolution):
    return mortonEncode(linearToGrid(linearIndex, gridResolution))

def getDenseCellOffset(baseResolution, level):
    baseX = baseResolution[0].int()
    baseY = baseResolution[1].int()
    offset = 0
    for l in range(1, level):
        levelRes = baseX * baseY
        offset += levelRes
        baseX >>= 1
        baseY >>= 1
    return offset