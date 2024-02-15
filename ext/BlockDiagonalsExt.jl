module BlockDiagonalsExt

import ProbNumDiffEq: ProbNumDiffEqBlockDiagonal, blocks
import BlockDiagonals: BlockDiagonal

BlockDiagonal(M::ProbNumDiffEqBlockDiagonal) = BlockDiagonal(blocks(M))

end
