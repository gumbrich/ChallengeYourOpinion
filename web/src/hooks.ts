import { useCallback, useEffect, useState } from 'react';
import type { ZodType } from 'zod';
export function useResource<T>(loader: (signal: AbortSignal) => Promise<T>) {
  const [data, setData] = useState<T>();
  const [error, setError] = useState(false);
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    setError(false);
    setData(undefined);
    loader(controller.signal)
      .then((value) => {
        if (!controller.signal.aborted) setData(value);
      })
      .catch(() => {
        if (!controller.signal.aborted) setError(true);
      });
    return () => controller.abort();
  }, [loader, attempt]);
  return { data, error, retry: () => setAttempt((v) => v + 1) };
}
export function useLocalState<T>(key: string, initial: T, schema: ZodType<T>) {
  const [value, setValue] = useState<T>(() => {
    try {
      const raw = localStorage.getItem(key);
      return raw ? schema.parse(JSON.parse(raw)) : initial;
    } catch {
      return initial;
    }
  });
  const [failed, setFailed] = useState(false);
  const update = useCallback((next: T | ((previous: T) => T)) => {
    setValue((previous) => (typeof next === 'function' ? (next as (p: T) => T)(previous) : next));
  }, []);
  useEffect(() => {
    try {
      localStorage.setItem(key, JSON.stringify(value));
      setFailed(false);
    } catch {
      setFailed(true);
    }
  }, [key, value]);
  return [value, update, failed] as const;
}
