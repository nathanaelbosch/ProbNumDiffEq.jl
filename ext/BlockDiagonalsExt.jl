module BlockDiagonalsExt

import ProbNumDiffEq: ProbNumDiffEqBlockDiagonal, blocks
using BlockDiagonals

BlockDiagonal(M::ProbNumDiffEqBlockDiagonal) = BlockDiagonal(blocks(M))

end
