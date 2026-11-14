/**
 * @module SYS-KERNEL/opfs-blobs
 * Origin Private File System (OPFS) adapter implementing `BlobBackend` (`opfs://blobs/<sha256>`).
 * Writes to a temporary staging file or flushes/closes transactional `FileSystemWritableFileStream`
 * so failed or quota-exceeded writes never leave partial blobs visible.
 * Cyclomatic complexity V(G) <= 10 per function.
 */
import type { Sha256Hex } from "../protocol/types.ts";
import { type BlobBackend, StorageQuotaError } from "./blobs.ts";

export const OPFS_BLOBS_DIR = "blobs" as const;
const SHA256_HEX_RE = /^[0-9a-f]{64}$/;

export interface OpfsWritableFileStreamLike {
  write(data: Uint8Array): Promise<void>;
  close(): Promise<void>;
  abort?(reason?: unknown): Promise<void>;
}

export interface OpfsFileLike {
  arrayBuffer(): Promise<ArrayBuffer>;
}

export interface OpfsFileHandleLike {
  getFile(): Promise<OpfsFileLike>;
  createWritable(options?: {
    keepExistingData?: boolean;
  }): Promise<OpfsWritableFileStreamLike>;
}

export interface OpfsDirectoryHandleLike {
  getDirectoryHandle(
    name: string,
    options?: { create?: boolean },
  ): Promise<OpfsDirectoryHandleLike>;
  getFileHandle(
    name: string,
    options?: { create?: boolean },
  ): Promise<OpfsFileHandleLike>;
  removeEntry(name: string): Promise<void>;
  entries(): AsyncIterable<[string, { readonly kind: "file" | "directory" }]>;
}

export type OpfsRootProvider = () => Promise<OpfsDirectoryHandleLike>;

function isNotFoundError(err: unknown): boolean {
  if (!err || typeof err !== "object") return false;
  const e = err as { name?: string; code?: string | number };
  return e.name === "NotFoundError" || e.code === "ENOENT";
}

function isQuotaError(err: unknown): boolean {
  if (!err || typeof err !== "object") return false;
  const e = err as { name?: string; code?: string | number };
  return (
    e.name === "QuotaExceededError" ||
    e.code === "ERR_STORAGE_QUOTA_EXCEEDED" ||
    e.code === 22
  );
}

function normalizeOpfsError(err: unknown, fallbackMessage: string): Error {
  if (err instanceof StorageQuotaError) return err;
  if (isQuotaError(err)) {
    return new StorageQuotaError();
  }
  if (err instanceof Error) return err;
  return new Error(fallbackMessage);
}

function assertValidSha256Hex(sha256: string): void {
  if (!SHA256_HEX_RE.test(sha256)) {
    throw new Error(`ERR_INVALID_BLOB_HASH: ${sha256}`);
  }
}

function defaultOpfsRootProvider(): Promise<OpfsDirectoryHandleLike> {
  const nav = globalThis.navigator as
    | {
        storage?: {
          getDirectory?: () => Promise<OpfsDirectoryHandleLike>;
        };
      }
    | undefined;
  if (!nav?.storage?.getDirectory) {
    return Promise.reject(
      new Error(
        "ERR_OPFS_UNAVAILABLE: navigator.storage.getDirectory is not available",
      ),
    );
  }
  return nav.storage.getDirectory();
}

export class OpfsBlobBackend implements BlobBackend {
  private readonly getRoot: OpfsRootProvider;
  private blobsDirPromise: Promise<OpfsDirectoryHandleLike> | null = null;

  constructor(rootProvider: OpfsRootProvider = defaultOpfsRootProvider) {
    this.getRoot = rootProvider;
  }

  private getBlobsDirectory(): Promise<OpfsDirectoryHandleLike> {
    if (this.blobsDirPromise !== null) return this.blobsDirPromise;
    this.blobsDirPromise = this.getRoot()
      .then((root) => root.getDirectoryHandle(OPFS_BLOBS_DIR, { create: true }))
      .catch((err) => {
        this.blobsDirPromise = null;
        throw normalizeOpfsError(err, "ERR_OPFS_DIR_INIT_FAILED");
      });
    return this.blobsDirPromise;
  }

  public async readBytes(sha256: Sha256Hex): Promise<Uint8Array | null> {
    assertValidSha256Hex(sha256);
    const dir = await this.getBlobsDirectory();
    try {
      const handle = await dir.getFileHandle(sha256, { create: false });
      const file = await handle.getFile();
      const buffer = await file.arrayBuffer();
      return new Uint8Array(buffer);
    } catch (err) {
      if (isNotFoundError(err)) return null;
      throw normalizeOpfsError(err, `ERR_OPFS_READ_FAILED: ${sha256}`);
    }
  }

  public async writeBytes(sha256: Sha256Hex, data: Uint8Array): Promise<void> {
    assertValidSha256Hex(sha256);
    const existing = await this.readBytes(sha256);
    if (existing !== null) return;

    const dir = await this.getBlobsDirectory();
    const stagingName = `${sha256}.tmp`;
    let writable: OpfsWritableFileStreamLike | null = null;

    try {
      // Stage into a temporary file first so quota failure or mid-stream error
      // cannot leave a partial target blob at <sha256>.
      const stagingHandle = await dir.getFileHandle(stagingName, { create: true });
      writable = await stagingHandle.createWritable({ keepExistingData: false });
      await writable.write(new Uint8Array(data));
      await writable.close();
      writable = null;

      // Publish to canonical <sha256> after staging succeeds, then remove staging file.
      const finalHandle = await dir.getFileHandle(sha256, { create: true });
      writable = await finalHandle.createWritable({ keepExistingData: false });
      await writable.write(new Uint8Array(data));
      await writable.close();
      writable = null;
      await dir.removeEntry(stagingName).catch(() => {});
    } catch (err) {
      if (writable?.abort) {
        await writable.abort(err).catch(() => {});
      }
      await dir.removeEntry(stagingName).catch(() => {});
      await dir.removeEntry(sha256).catch(() => {});
      throw normalizeOpfsError(err, `ERR_OPFS_WRITE_FAILED: ${sha256}`);
    }
  }

  public async listHashes(): Promise<Sha256Hex[]> {
    const dir = await this.getBlobsDirectory();
    const hashes: Sha256Hex[] = [];
    for await (const [name, handle] of dir.entries()) {
      if (handle.kind === "file" && SHA256_HEX_RE.test(name)) {
        hashes.push(name);
      }
    }
    hashes.sort();
    return hashes;
  }
}
