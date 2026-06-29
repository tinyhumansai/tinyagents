# Graph Error Handling And Fault Tolerance

Default behavior:

- node error fails the run
- retry only if policy allows it
- timeout fails the node
- checkpoint remains at the last completed boundary
- pending writes from completed tasks remain available when the checkpointer
  supports them

Target behavior:

- route node errors to a node-specific or default error handler
- expose handled errors in task events and checkpoint task metadata
- retry with backoff and jitter
- support optional nodes by explicit policy
- support cooperative drain/shutdown with a drain reason
- distinguish graph recursion errors from node failures

Error types should distinguish:

- missing start
- duplicate node
- missing node
- missing edge target
- duplicate route
- missing route
- invalid command target
- invalid parent graph command
- invalid send target
- recursion limit
- checkpoint required
- checkpoint missing
- interrupt resume mismatch
- reducer conflict
- invalid concurrent update
- node timeout
- node failure
- graph drained
- serialization failure
- checkpoint backend failure
