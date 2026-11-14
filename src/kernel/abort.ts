/** Await cancellation without trusting an external adapter to settle. */
export function awaitWithAbort<T>(operation: PromiseLike<T>, signal: AbortSignal): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const cleanup = () => signal.removeEventListener("abort", abort);
    const abort = () => { cleanup(); reject(signal.reason ?? new DOMException("Run stopped", "AbortError")); };
    signal.addEventListener("abort", abort, { once: true });
    Promise.resolve(operation).then(
      (value) => { cleanup(); resolve(value); },
      (error) => { cleanup(); reject(error); },
    );
    if (signal.aborted) abort();
  });
}
