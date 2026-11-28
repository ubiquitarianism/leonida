; Synthetic SPIR-V harness for the translator's descriptor lookup function.
; One workgroup per key. The table and keys share binding 2. Return hash and index in the heap.
OpCapability Shader
OpCapability Int64
OpCapability PhysicalStorageBufferAddresses
OpExtension "SPV_KHR_physical_storage_buffer"
OpMemoryModel PhysicalStorageBuffer64 GLSL450
OpEntryPoint GLCompute %main "main" %group %desc_map %pc
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %group BuiltIn WorkgroupId
OpDecorate %words ArrayStride 4
OpMemberDecorate %table 0 Offset 0
OpDecorate %table Block
OpDecorate %desc_map DescriptorSet 0
OpDecorate %desc_map Binding 2
OpMemberDecorate %push 0 Offset 128
OpDecorate %push Block
%void = OpTypeVoid
%bool = OpTypeBool
%uint = OpTypeInt 32 0
%ulong = OpTypeInt 64 0
%v3uint = OpTypeVector %uint 3
%true = OpConstantTrue %bool
%words = OpTypeRuntimeArray %uint
%table = OpTypeStruct %words
%push = OpTypeStruct %ulong
%table_ptr = OpTypePointer StorageBuffer %table
%group_ptr = OpTypePointer Input %v3uint
%push_ptr = OpTypePointer PushConstant %push
%heap_ptr = OpTypePointer PushConstant %ulong
%physical_word = OpTypePointer PhysicalStorageBuffer %uint
%_ptr_Function_uint = OpTypePointer Function %uint
%_ptr_SB_uint = OpTypePointer StorageBuffer %uint
%desc_map = OpVariable %table_ptr StorageBuffer
%pc = OpVariable %push_ptr PushConstant
%group = OpVariable %group_ptr Input
%entry_type = OpTypeFunction %void
%fn_desc_lookup = OpTypeFunction %uint %uint %uint %uint %uint %uint %uint %uint %uint
; CONSTANTS
%main = OpFunction %void None %entry_type
%entry = OpLabel
%xyz = OpLoad %v3uint %group
%i = OpCompositeExtract %uint %xyz 0
%row = OpIMul %uint %i %c8
%start = OpIAdd %uint %row %c1401
; LOOKUP_AND_READBACK
OpReturn
OpFunctionEnd
; LOOKUP_FUNCTION
