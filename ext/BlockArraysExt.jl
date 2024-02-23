module BlockArraysExt

import ProbNumDiffEq: BlocksOfDiagonals
import BlockArrays: BlockArray, mortar
import LinearAlgebra: Diagonal

BlockArray(M::BlocksOfDiagonals) = begin
    d = length(M.blocks)
    a, b = size(M.blocks[1])
    return mortar([Diagonal([M.blocks[k][i, j] for k in 1:d]) for i in 1:a, j in 1:b])
end

end
