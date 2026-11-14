/**
 * @module SYS-KERNEL/blobs
 * Content-addressed binary/text blob store (OPFS in browser, InMemory/Quota-Sim in tests).
 * All files are stored immutable under their lowercase hex SHA-256 digest.
 */
import type { Sha256Hex } from "../protocol/types.ts";

export class StorageQuotaError extends Error {
  public readonly code = "ERR_STORAGE_QUOTA_EXCEEDED" as const;
  constructor(message = "ERR_STORAGE_QUOTA_EXCEEDED: Browser storage quota exceeded or denied") {
    super(message);
    this.name = "StorageQuotaError";
  }
}

export interface BlobBackend {
  readBytes(sha256: Sha256Hex): Promise<Uint8Array | null>;
  writeBytes(sha256: Sha256Hex, data: Uint8Array): Promise<void>;
  listHashes(): Promise<Sha256Hex[]>;
}

/**
 * Compute lowercase hex SHA-256 using standard Web Crypto (`crypto.subtle`).
 */
export async function computeSha256Hex(data: Uint8Array): Promise<Sha256Hex> {
  const digestBuffer = await globalThis.crypto.subtle.digest(
    "SHA-256",
    data as unknown as BufferSource,
  );
  const bytes = new Uint8Array(digestBuffer);
  let hex = "";
  for (let i = 0; i < bytes.length; i++) {
    hex += bytes[i]!.toString(16).padStart(2, "0");
  }
  return hex;
}

export function bytesToBase64(bytes: Uint8Array): string {
  let binary = "";
  for (let i = 0; i < bytes.byteLength; i++) {
    binary += String.fromCharCode(bytes[i]!);
  }
  return globalThis.btoa(binary);
}

export function base64ToBytes(b64: string): Uint8Array {
  const binary = globalThis.atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}

export class ContentAddressedBlobStore {
  private readonly backend: BlobBackend;

  constructor(backend: BlobBackend) {
    this.backend = backend;
  }

  public async putText(content: string): Promise<Sha256Hex> {
    const bytes = new TextEncoder().encode(content);
    return this.putBytes(bytes);
  }

  public async putBytes(data: Uint8Array): Promise<Sha256Hex> {
    const sha256 = await computeSha256Hex(data);
    const existing = await this.backend.readBytes(sha256);
    if (existing !== null) {
      return sha256;
    }
    await this.backend.writeBytes(sha256, new Uint8Array(data));
    return sha256;
  }

  public async getBytesVerified(expectedSha256: Sha256Hex): Promise<Uint8Array> {
    const raw = await this.backend.readBytes(expectedSha256);
    if (raw === null) {
      throw new Error(`ERR_BLOB_MISSING: No blob found for sha256=${expectedSha256}`);
    }
    const actualSha256 = await computeSha256Hex(raw);
    if (actualSha256 !== expectedSha256) {
      throw new Error(
        `ERR_BLOB_CORRUPTED: Expected ${expectedSha256}, computed ${actualSha256}`,
      );
    }
    return raw;
  }

  public async getTextVerified(expectedSha256: Sha256Hex): Promise<string> {
    const bytes = await this.getBytesVerified(expectedSha256);
    return new TextDecoder().decode(bytes);
  }

  public async exportAllBlobs(): Promise<Record<Sha256Hex, string>> {
    const hashes = await this.backend.listHashes();
    return this.exportBlobs(hashes);
  }

  public async exportBlobs(
    hashes: Iterable<Sha256Hex>,
  ): Promise<Record<Sha256Hex, string>> {
    const uniqueSorted = Array.from(new Set(hashes)).sort();
    const out: Record<Sha256Hex, string> = {};
    for (const h of uniqueSorted) {
      const bytes = await this.getBytesVerified(h);
      out[h] = bytesToBase64(bytes);
    }
    return out;
  }
}

/**
 * In-memory OPFS stand-in with configurable byte quota for Scenario 6 testing.
 */
export class InMemoryBlobBackend implements BlobBackend {
  private readonly blobs = new Map<Sha256Hex, Uint8Array>();
  private maxTotalBytes: number;
  private currentBytes = 0;

  constructor(maxTotalBytes = Number.POSITIVE_INFINITY) {
    this.maxTotalBytes = maxTotalBytes;
  }

  public setMaxQuotaBytes(limitBytes: number): void {
    this.maxTotalBytes = limitBytes;
  }

  public async readBytes(sha256: Sha256Hex): Promise<Uint8Array | null> {
    const found = this.blobs.get(sha256);
    return found ? new Uint8Array(found) : null;
  }

  public async writeBytes(sha256: Sha256Hex, data: Uint8Array): Promise<void> {
    if (this.blobs.has(sha256)) return;
    if (this.currentBytes + data.byteLength > this.maxTotalBytes) {
      throw new StorageQuotaError();
    }
    this.blobs.set(sha256, new Uint8Array(data));
    this.currentBytes += data.byteLength;
  }

  public async listHashes(): Promise<Sha256Hex[]> {
    return Array.from(this.blobs.keys()).sort();
  }
}