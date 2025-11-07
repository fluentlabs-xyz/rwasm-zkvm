| Instruction |  Status|Comment|
| ------------| --|--|
| Unreachable  | &cross;  |   |
| Trap(TrapCode) | &cross; |  |
| LocalGet(LocalDepth)  |:white_check_mark: | |
|  LocalSet(LocalDepth)  | :white_check_mark: | |
|  LocalTee(LocalDepth)  | :white_check_mark: | |
|    Br(BranchOffset)  | :white_check_mark: | |
|    BrIfEqz(BranchOffset)  | :white_check_mark: | |
|    BrIfNez(BranchOffset)  | :white_check_mark: | |
|    BrTable(BranchTableTargets)  | :white_check_mark: | |
|   ConsumeFuel(BlockFuel)  | &cross; |  |
|   ConsumeFuelStack  | &cross; |  |
|  Return  | :white_check_mark: | |
| ReturnCallInternal(CompiledFunc) | &cross; |  |
|   ReturnCall(SysFuncIdx)| &cross; |  |
|  ReturnCallIndirect(SignatureIdx) | &cross; |  |
|   CallInternal(CompiledFunc) = | :white_check_mark: |
|  Call(SysFuncIdx)  | &cross; | Yao TODO |
|  CallIndirect(SignatureIdx)  | :white_check_mark: |
|  SignatureCheck(SignatureIdx) | &cross; |  |
|    StackCheck(MaxStackHeight) | &cross; |  |
|    RefFunc(CompiledFunc) | &cross; |  |
|    I32Const(UntypedValue)  | :white_check_mark: |
|  Drop = 0x62 | &cross; |  |
|   Select = 0x63 | &cross; |  |
|  GlobalGet(GlobalIdx)  | &cross; |  |
|   GlobalSet(GlobalIdx) | &cross; |  |
| I32Load(AddressOffset) | :white_check_mark: |
|    I32Load8S(AddressOffset) | :white_check_mark: |
|    I32Load8U(AddressOffset) | :white_check_mark: |
|    I32Load16S(AddressOffset) | :white_check_mark: |
|    I32Load16U(AddressOffset) | :white_check_mark: |
|    I32Store(AddressOffset) | :white_check_mark: |
|    I32Store8(AddressOffset) | :white_check_mark: |
|    I32Store16(AddressOffset) | :white_check_mark: |
|  MemorySize  | &cross; |  |
|    MemoryGrow  | &cross; |  |
|    MemoryFill  | &cross; |  |
|    MemoryCopy  | &cross; |  |
|    MemoryInit(DataSegmentIdx)  | &cross; |  |
|    DataDrop(DataSegmentIdx)  | &cross; |  |
|    TableSize(TableIdx)  | &cross; |  |
|   TableGrow(TableIdx)  | :white_check_mark: |
|   TableFill(TableIdx)   | &cross; | Alexi TODO |
|   TableGet(TableIdx)  | &cross; | Alexi TODO |
|    TableSet(TableIdx)  | &cross; | Alexi TODO |
|  TableCopy(TableIdx, TableIdx) =  | &cross; | Alexi TODO |
|    TableInit(ElementSegmentIdx) =| :white_check_mark: |
|    ElemDrop(ElementSegmentIdx) =  | &cross; | Alexi TODO |
|I32Eqz  | :white_check_mark: |
|    I32Eq  | :white_check_mark: |
|    I32Ne  | :white_check_mark: |
|    I32LtS  | :white_check_mark: |
|    I32LtU  | :white_check_mark: |
|    I32GtS  | :white_check_mark: |
|    I32GtU | :white_check_mark: |
|    I32LeS  | :white_check_mark: |
|    I32LeU | :white_check_mark: |
|    I32GeS  | :white_check_mark: |
|    I32GeU  | :white_check_mark: |
|    I32Clz | :white_check_mark: |
|    I32Ctz  | :white_check_mark: |
|    I32Popcnt  | :white_check_mark: |
|    I32Add  | :white_check_mark: |
|    I32Sub  | :white_check_mark: |
|    I32Mul  | :white_check_mark: |
|    I32DivS  | :white_check_mark: |
|    I32DivU  | :white_check_mark: |
|    I32RemS  | :white_check_mark: |
|    I32RemU  | :white_check_mark: |
|    I32And  | :white_check_mark: |
|    I32Or  | :white_check_mark: |
|    I32Xor  | :white_check_mark: |
|    I32Shl  | :white_check_mark: |
|    I32ShrS  | :white_check_mark: |
|    I32ShrU  | :white_check_mark: |
|    I32Rotl  | :white_check_mark: |
|    I32Rotr  | :white_check_mark: |
|    I32WrapI64  | &cross; | Sulyiman TODO |
|    I32Extend8S  | &cross; |  |
|    I32Extend16S   | &cross; |  |
 |   I32Mul64  | &cross; | Sulyiman TODO |
 |   I32Add64  | &cross; | Sulyiman TODO |