from abc import ABC, abstractmethod

from .block import Block
from ct_autodiff.utils.formating import box_text


class Workflow(ABC):
    def __init__(self):
        """
        Initializes a workflow with a name and an empty list of blocks.
        
        :param name: A unique identifier for the workflow.
        :type name: str
        """
        self.name = self.__class__.__name__
        self.blocks = {}

    def add_block(self, block: Block):
        """
        Adds a block to the workflow. Important note, all blocks must be of type Block,
        otherwise a ValueError is raised. Also note that the order of blocks matters,
        as the output of one block is the input to the next block in the workflow.

        This is the prefered way to add blocks to a workflow,
        as it allows for validation of the block type and ensures the correct order of execution.
        
        :param block: The block to be added to the workflow.
        :type block: Block
        :raises ValueError: If the block is not an instance of Block.
        """
        if not isinstance(block, Block):
            raise ValueError(f"All blocks must be of type Block, got {type(block)}")
        self.blocks[block.name] = block

    def run(self, input_data):
        """
        Executes the workflow by sequentially running each block on the output of the previous block.
        
        :param input_data: The initial input data for the workflow, which will be passed to the first block.
        :type input_data: Any
        :return: The output of the final block in the workflow after processing the input data through
                 all blocks.
        :rtype: Any
        """
        out = input_data
        for block in self.blocks.values():
            out = block.execute(out)
        return out
    
    def parameters(self):
        """
        Yields the parameters of all blocks in the workflow.

        :return: An iterator over the parameters of all blocks.
        :rtype: Iterator
        """
        for block in self.blocks.values():
            yield from block.parameters()

    def to(self, device):
        """Moves all blocks in the workflow to the specified device."""
        for block in self.blocks.values():
            block.to(device)
        return self
    
    def __repr__(self):
        return f"Workflow(name={self.name}, blocks={[b.name for b in self.blocks.values()]})"
    
    def __str__(self):
        if not self.blocks:
            return f"{self.name}: <no blocks>"

        lines = [f"{self.name}:"]
        for i, block in enumerate(self.blocks.values()):
            lines.append(f"{block}")
            if i != len(self.blocks) - 1:
                # This is just to make it look prety in a print statement,
                # it adds an arrow between blocks to indicate the flow of data
                lines.append("   │")
                lines.append("   ▼")
        return "\n".join(lines)
    