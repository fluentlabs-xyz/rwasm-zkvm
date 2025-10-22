| Instruction | Type | Status|V4Exec|V4Circuit|Comment|
| ------------|-------------|--|--|--|--|
| LocalGet(LocalDepth) |Local| :heavy_check_mark:|||
| LocalSet(LocalDepth) |Local| :heavy_check_mark:|||
LocalTee(LocalDepth) |Local| :heavy_check_mark:|||
Br(BranchOffset),|Branching| :heavy_check_mark:|:heavy_check_mark:||
BrIfEqz(BranchOffset),|Branching | :heavy_check_mark:|:heavy_check_mark:||
BrIfNez(BranchOffset),|Branching | :heavy_check_mark:|:heavy_check_mark:||
BrAdjust(BranchOffset), // without dropkeep => drop ||||
BrAdjustIfNez(BranchOffset), // without dropkeep => drop|||
BrTable(BranchTableTargets), // drop? how? |||
Unreachable,| |  |
ConsumeFuel(BlockFuel),| |  |
Return(DropKeep), |Call| WIP|
ReturnIfNez(DropKeep), // not sure| |  |
ReturnCallInternal(CompiledFunc), // not sure| |  |
ReturnCall(FuncIdx), // not sure| |  |
ReturnCallIndirect(SignatureIdx), // not sure| |  |
CallInternal(CompiledFunc), |Call|WIP|
Call(FuncIdx),| |  |
CallIndirect(SignatureIdx), // not sure| |  |
SignatureCheck(SignatureIdx),| |  |
Drop,| |  |
Select,| |  |
GlobalGet(GlobalIdx),| |  |
GlobalSet(GlobalIdx),| |  |
I32Load(AddressOffset), | Memory| :heavy_check_mark:|:heavy_check_mark:||
I32Load8S(AddressOffset),| Memory| :heavy_check_mark:|:heavy_check_mark:||
I32Load8U(AddressOffset),| Memory| :heavy_check_mark:|:heavy_check_mark:||
I32Load16S(AddressOffset),| Memory| :heavy_check_mark:|:heavy_check_mark:||
I32Load16U(AddressOffset),| Memory| :heavy_check_mark:|:heavy_check_mark:||
I32Store(AddressOffset),| Memory| :heavy_check_mark:|:heavy_check_mark:||
I32Store8(AddressOffset),| Memory| :heavy_check_mark:|:heavy_check_mark:||
I32Store16(AddressOffset),| Memory| :heavy_check_mark:|:heavy_check_mark:||
MemorySize,| |  |
MemoryGrow,| |  |
MemoryFill, // precompile?| |  |
MemoryCopy, // precompile?| |  |
MemoryInit(DataSegmentIdx),| |  |
DataDrop(DataSegmentIdx),| |  |
TableSize(TableIdx), // not sure| |  |
TableGrow(TableIdx), // not sure| |  |
TableFill(TableIdx), // not sure| |  |
TableGet(TableIdx), // not sure| |  |
TableSet(TableIdx), // not sure| |  |
TableCopy(TableIdx), // not sure| |  |
TableInit(ElementSegmentIdx), // not sure| |  |
ElemDrop(ElementSegmentIdx), // not sure| |  |
RefFunc(FuncIdx),| |  |
I32Const(UntypedValue),|Branching | :heavy_check_mark:|
I32Eqz,|Arithmetic | :heavy_check_mark:|:heavy_check_mark:||
I32Eq,|Arithmetic | :heavy_check_mark:|:heavy_check_mark:||
I32Ne,|Arithmetic | :heavy_check_mark:|:heavy_check_mark:||
I32LtS,|Arithmetic | :heavy_check_mark:|:heavy_check_mark:||
I32LtU,|Arithmetic | :heavy_check_mark:|:heavy_check_mark:||
I32GtS,|Arithmetic | :heavy_check_mark:|:heavy_check_mark:||
I32GtU,|Arithmetic | :heavy_check_mark:|:heavy_check_mark:||
I32LeS,|Arithmetic | :heavy_check_mark:|:heavy_check_mark:||
I32LeU,|Arithmetic | :heavy_check_mark:|:heavy_check_mark:||
I32GeS,|Arithmetic | :heavy_check_mark:|:heavy_check_mark:||
I32GeU,|Arithmetic | :heavy_check_mark:|:heavy_check_mark:||
I32Clz, // replace with snippet| |  |
I32Ctz, // replace with snippet| |  |
I32Popcnt, // replace with snippet| |  |""""""""""""
I32Add,|Arithmetic | :heavy_check_mark:|:heavy_check_mark:||
I32Sub,|Arithmetic | :heavy_check_mark:|:heavy_check_mark:||
I32Mul,|Arithmetic | :heavy_check_mark:|:heavy_check_mark:||
I32DivS,|Arithmetic | :heavy_check_mark:|:heavy_check_mark:||
I32DivU,|Arithmetic | :heavy_check_mark:|:heavy_check_mark:||
I32RemS,|Arithmetic | :heavy_check_mark:|:heavy_check_mark:||
I32RemU,|Arithmetic | :heavy_check_mark:|:heavy_check_mark:||
I32And,|Arithmetic | :heavy_check_mark:|:heavy_check_mark:||
I32Or,|Arithmetic | :heavy_check_mark:|:heavy_check_mark:||
I32Xor,|Arithmetic | :heavy_check_mark:|:heavy_check_mark:||
I32Shl,|Arithmetic | :heavy_check_mark:|:heavy_check_mark:||
I32ShrS,|Arithmetic | :heavy_check_mark:|:heavy_check_mark:||
I32ShrU,|Arithmetic | :heavy_check_mark:|:heavy_check_mark:||
I32Rotl,|Arithmetic | :x:|
I32Rotr,|Arithmetic | :x:|
I32Extend8S,|Arithmetic|:x:|
I32Extend16S,|Arithmetic|:x:|
