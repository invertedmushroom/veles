/**
 * @module SYS-KERNEL/lock
 * Multi-tab single-writer coordination (!INVARIANT [TP-KERN-01]).
 * Grants exclusive write authority to the first tab/worker holding the workspace lock;
 * secondary tabs enter ReadOnly_Observer mode.
 */
import type { WorkspaceId } from "../protocol/types.ts";

export type TabAccessMode = "ReadWrite_Leader" | "ReadOnly_Observer";

export interface WorkspaceLease {
  readonly workspaceId: WorkspaceId;
  readonly holderId: string;
  readonly mode: TabAccessMode;
  release(): Promise<void>;
}

export interface LockProvider {
  acquireWorkspaceLease(
    workspaceId: WorkspaceId,
    holderId: string,
  ): Promise<WorkspaceLease>;
}

/**
 * Deterministic in-process lock registry for multi-tab simulation in unit/scenario tests
 * and fallback when navigator.locks is unavailable.
 */
export class InMemoryLockProvider implements LockProvider {
  private readonly activeHolders = new Map<WorkspaceId, string>();

  public async acquireWorkspaceLease(
    workspaceId: WorkspaceId,
    holderId: string,
  ): Promise<WorkspaceLease> {
    const current = this.activeHolders.get(workspaceId);
    if (current !== undefined && current !== holderId) {
      return {
        workspaceId,
        holderId,
        mode: "ReadOnly_Observer",
        release: async () => {},
      };
    }

    this.activeHolders.set(workspaceId, holderId);
    return {
      workspaceId,
      holderId,
      mode: "ReadWrite_Leader",
      release: async () => {
        if (this.activeHolders.get(workspaceId) === holderId) {
          this.activeHolders.delete(workspaceId);
        }
      },
    };
  }
}

/**
 * Browser Web Locks API adapter (`navigator.locks.request(..., { ifAvailable: true })`).
 */
export class BrowserNavigatorLockProvider implements LockProvider {
  public async acquireWorkspaceLease(
    workspaceId: WorkspaceId,
    holderId: string,
  ): Promise<WorkspaceLease> {
    const nav = globalThis.navigator as
      | {
          locks?: {
            request: (
              name: string,
              options: { mode: "exclusive"; ifAvailable: true },
              callback: (lock: unknown | null) => Promise<void>,
            ) => Promise<void>;
          };
        }
      | undefined;

    if (!nav?.locks) {
      throw new Error("ERR_WEB_LOCKS_UNAVAILABLE: navigator.locks is required for browser coordination");
    }

    let releaseLock!: () => void;
    const lockReleasedPromise = new Promise<void>((resolve) => {
      releaseLock = resolve;
    });

    const modePromise = new Promise<TabAccessMode>((resolve) => {
      void nav.locks!.request(
        `workspace:${workspaceId}`,
        { mode: "exclusive", ifAvailable: true },
        async (lock) => {
          if (lock === null) {
            resolve("ReadOnly_Observer");
            return;
          }
          resolve("ReadWrite_Leader");
          await lockReleasedPromise;
        },
      );
    });

    const mode = await modePromise;
    return {
      workspaceId,
      holderId,
      mode,
      release: async () => {
        releaseLock();
      },
    };
  }
}