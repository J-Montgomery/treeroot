# Treeroot

Many codebases ultimately devolve into a tortured nest of if-else logic. Treeroot is an exploration of how far I can push all of that logic elsewhere.

Treeroot is, conceptually, an in-memory database designed to replace imperative branching with declarative queries.

Instead of writing loops to check the state of objects, you treat your data as a collection of number-like facts (e.g. sensor values, entity positions, or state flags). 

Treeroot processes queries about this data with massive parallelism, using compile-time query planning where possible. Simple queries might execute in nanoseconds per elements. Complex queries can be written in a monadic datalog that's powerful enough for most use cases while maintaining high performance.