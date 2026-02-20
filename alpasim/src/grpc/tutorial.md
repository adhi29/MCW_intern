# GRPC tutorial
This is meant as a tutorial for implementing gRPC (google remote procedure call) endpoints for the alpamayo simulator.

## Preliminaries
gRPC combines 
1. A language-agnostic binary serialization/deserialization format (`protobuf`)
2. A language-agnostic way to describe services (without implementation)
3. A bunch of packages for different languages which can generate code to serialize/deserialize the messages and provide implementation stubs (think `C` headers) for serving/consuming the APIs.

The format for describing messages and services is `.proto`. The package for serving/consuming `gRPC` is called `grpcio` in pip and imported as `import grpc`. The package for autogenerating code is called `grpcio-tools` on pip and is imported as `grpc_tools` in Python. You do not need the latter in your code.

## The role of this project
This module contains the `.proto` files defining all network interfaces in alpasim and code
(`compile_protos.py`) which will use `grpcio-tools` to build these into Python "headers" (empty
base classes). All you should need to do is `uv run compile-protos` and include this package as
editable.

## Generated files
For a given `<some_name>.proto` file the compiler will create 3 files: `<some_name>_pb2.py`, `<some_name>_pb2_grpc.py`, and `<some_name>_pb2.pyi`.

The `*_pb2.py` files contain message (struct) definitions for the serialization format - think `dataclass` but completely unreadable. `*_pb2.pyi` provides type hints for those, making your IDE actually helpful. When building your service, you will receive these structs as inputs and return them as outputs.

The `*_pb2_grpc.py` files contain the "headers" for the service itself and unfortunately comes without `.pyi` hints. This file will contain 3 objects of interest. On the example of a `runtime.proto` file defining the following service
```proto
service RuntimeService {
    rpc simulate (SimulationRequest) returns (SimulationReturn);
}
```
the generated `runtime_pb2_grpc.py` will contain
1. `class RuntimeServiceStub`
2. `class RuntimeServiceServicer`
3. `def add_RuntimeServiceServicer_to_server`

Number 1. is for **clients**, 2. and 3. are for the **server**.

### Directory structure
Unfortunately, the bare gRPC codegen doesn't produce valid Python packages - without `__init__.py` and relative imports you can't just output the generated files to an arbitrary location in your codebase and import them from there. This is the reason for this package, which has a well defined root, contains "hand-made" `__init__.py` and after installing allows imports like `from alpasim_grpc.v0.your_service_pb2_grpc import your_thing`.

## Implementing
For more information on implementation, see the [gRPC official docs](https://grpc.io/docs/languages/python/basics/).
