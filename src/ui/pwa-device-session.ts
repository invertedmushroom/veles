/**
 * @module SYS-UI/pwa-device-session
 * Browser PWA installation, prompted Service Worker updates (`virtual:pwa-register`),
 * Screen Wake Lock (`navigator.wakeLock`) with `visibilitychange` restoration,
 * native document fullscreen, and durable storage persistence (`navigator.storage.persist`).
 * Zero imports from `src/kernel/**` (`!INVARIANT [Coupling_Cohesion_Law]`).
 * Cyclomatic complexity V(G) <= 10 per function.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { registerSW } from "virtual:pwa-register";

interface InstallChoice {
  readonly outcome: "accepted" | "dismissed";
  readonly platform: string;
}

export interface BeforeInstallPromptEvent extends Event {
  prompt: () => Promise<void>;
  userChoice: Promise<InstallChoice>;
}

export interface ScreenWakeLockSentinel extends EventTarget {
  readonly released: boolean;
  release: () => Promise<void>;
}

type WakeLockNavigator = Navigator & {
  readonly wakeLock?: {
    request: (type: "screen") => Promise<ScreenWakeLockSentinel>;
  };
};

export interface PwaDeviceSessionView {
  readonly displayMode: "installed app" | "browser tab";
  readonly isStandalone: boolean;
  readonly networkOnline: boolean;
  readonly canInstall: boolean;
  readonly installButtonLabel: string;
  readonly updateAvailable: boolean;
  readonly applyingUpdate: boolean;
  readonly pwaStatus: string;
  readonly wakeLockSupported: boolean;
  readonly wakeLockActive: boolean;
  readonly wakeLockPending: boolean;
  readonly wakeLockButtonLabel: string;
  readonly wakeLockStatus: string;
  readonly fullscreenActive: boolean;
  readonly fullscreenButtonLabel: string;
  readonly persisted: boolean;
  readonly triggerInstall: () => Promise<void>;
  readonly triggerUpdate: () => void;
  readonly toggleWakeLock: () => Promise<void>;
  readonly toggleAppFullscreen: () => Promise<void>;
  readonly requestPersistence: () => Promise<void>;
}

function formatErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function computeInstallButtonLabel(
  isStandalone: boolean,
  hasPrompt: boolean,
): string {
  if (isStandalone) return "Installed";
  return hasPrompt ? "Install app" : "Install via browser";
}

function computeWakeLockButtonLabel(
  active: boolean,
  pending: boolean,
  desired: boolean,
): string {
  if (active) return "Allow screen sleep";
  if (pending) return "Requesting…";
  return desired ? "Restore screen wake" : "Keep screen awake";
}

function isWakeLockSupported(): boolean {
  return (
    typeof navigator !== "undefined" &&
    (navigator as WakeLockNavigator).wakeLock !== undefined
  );
}

function readInitialStandalone(): boolean {
  return typeof window !== "undefined"
    ? window.matchMedia("(display-mode: standalone)").matches
    : false;
}

function readInitialOnline(): boolean {
  return typeof navigator !== "undefined" ? navigator.onLine : true;
}

function readInitialFullscreen(): boolean {
  return typeof document !== "undefined"
    ? Boolean(document.fullscreenElement)
    : false;
}

function readInitialPwaStatus(): string {
  return import.meta.env.PROD
    ? "Checking offline app shell…"
    : "Offline install is enabled in the built HTTPS preview, not the Vite dev server.";
}

function readInitialWakeLockStatus(): string {
  return isWakeLockSupported()
    ? "Screen wake lock is off."
    : "Screen Wake Lock is unavailable in this browser.";
}

function canAcquireScreenWakeLock(
  destroyed: boolean,
  desired: boolean,
  activeSentinel: ScreenWakeLockSentinel | null,
  pending: boolean,
  nav: WakeLockNavigator,
): nav is Required<WakeLockNavigator> {
  if (destroyed || !desired || activeSentinel || pending) return false;
  return document.visibilityState === "visible" && nav.wakeLock !== undefined;
}

function resolveRegisteredSwMessage(
  registration: ServiceWorkerRegistration | undefined,
): string {
  if (!registration) return "Offline app shell registration was unavailable.";
  return navigator.serviceWorker?.controller
    ? "Offline app shell is ready. Remote LLM and search calls still need a network."
    : "Offline app shell registered; finishing the first cache download.";
}

export function usePwaDeviceSession(): PwaDeviceSessionView {
  const [isStandalone, setIsStandalone] = useState<boolean>(readInitialStandalone);
  const [networkOnline, setNetworkOnline] = useState<boolean>(readInitialOnline);
  const [installPrompt, setInstallPrompt] =
    useState<BeforeInstallPromptEvent | null>(null);
  const [updateAvailable, setUpdateAvailable] = useState(false);
  const [applyingUpdate, setApplyingUpdate] = useState(false);
  const [pwaStatus, setPwaStatus] = useState<string>(readInitialPwaStatus);
  const [wakeLockSupported, setWakeLockSupported] =
    useState<boolean>(isWakeLockSupported);
  const [wakeLockActive, setWakeLockActive] = useState(false);
  const [wakeLockDesired, setWakeLockDesired] = useState(false);
  const [wakeLockPending, setWakeLockPending] = useState(false);
  const [wakeLockStatus, setWakeLockStatus] = useState<string>(
    readInitialWakeLockStatus,
  );
  const [fullscreenActive, setFullscreenActive] =
    useState<boolean>(readInitialFullscreen);
  const [persisted, setPersisted] = useState(false);

  const destroyedRef = useRef(false);
  const wakeLockRef = useRef<ScreenWakeLockSentinel | null>(null);
  const wakeLockDesiredRef = useRef(false);
  const wakeLockPendingRef = useRef(false);
  const updateSwRef = useRef<((reloadPage?: boolean) => Promise<void>) | null>(
    null,
  );

  const attachSentinelRelease = useCallback(
    (sentinel: ScreenWakeLockSentinel) => {
      sentinel.addEventListener(
        "release",
        () => {
          if (wakeLockRef.current !== sentinel) return;
          wakeLockRef.current = null;
          setWakeLockActive(false);
          if (document.visibilityState === "visible") {
            wakeLockDesiredRef.current = false;
            setWakeLockDesired(false);
            setWakeLockStatus(
              "The browser or operating system released the wake lock.",
            );
          } else {
            setWakeLockStatus(
              "Wake lock paused while hidden; it will restore on return.",
            );
          }
        },
        { once: true },
      );
    },
    [],
  );

  const acquireWakeLock = useCallback(async () => {
    const nav = navigator as WakeLockNavigator;
    if (
      !canAcquireScreenWakeLock(
        destroyedRef.current,
        wakeLockDesiredRef.current,
        wakeLockRef.current,
        wakeLockPendingRef.current,
        nav,
      )
    ) {
      return;
    }
    wakeLockPendingRef.current = true;
    setWakeLockPending(true);
    setWakeLockStatus("Requesting a screen wake lock…");
    try {
      const sentinel = await nav.wakeLock.request("screen");
      if (destroyedRef.current || !wakeLockDesiredRef.current) {
        await sentinel.release();
        return;
      }
      wakeLockRef.current = sentinel;
      setWakeLockActive(true);
      setWakeLockStatus(
        "Screen will stay awake while Veles remains visible.",
      );
      attachSentinelRelease(sentinel);
    } catch (err) {
      wakeLockDesiredRef.current = false;
      setWakeLockDesired(false);
      setWakeLockStatus(
        `Could not keep the screen awake: ${formatErrorMessage(err)}`,
      );
    } finally {
      wakeLockPendingRef.current = false;
      setWakeLockPending(false);
    }
  }, [attachSentinelRelease]);

  const releaseWakeLock = useCallback(async () => {
    wakeLockDesiredRef.current = false;
    setWakeLockDesired(false);
    const sentinel = wakeLockRef.current;
    wakeLockRef.current = null;
    setWakeLockActive(false);
    setWakeLockStatus("Screen wake lock is off.");
    await sentinel?.release().catch(() => {});
  }, []);

  const toggleWakeLock = useCallback(async () => {
    if (wakeLockDesiredRef.current || wakeLockRef.current) {
      await releaseWakeLock();
      return;
    }
    wakeLockDesiredRef.current = true;
    setWakeLockDesired(true);
    await acquireWakeLock();
  }, [acquireWakeLock, releaseWakeLock]);

  const triggerInstall = useCallback(async () => {
    const promptEvent = installPrompt;
    if (!promptEvent) return;
    setInstallPrompt(null);
    await promptEvent.prompt();
    const choice = await promptEvent.userChoice;
    setPwaStatus(
      choice.outcome === "accepted"
        ? "Install accepted; the standalone Veles app is ready."
        : "Install dismissed; it remains available from the browser menu.",
    );
  }, [installPrompt]);

  const triggerUpdate = useCallback(() => {
    if (!updateAvailable) return;
    setApplyingUpdate(true);
    setPwaStatus("Applying update…");
    void updateSwRef.current?.(true);
  }, [updateAvailable]);

  const toggleAppFullscreen = useCallback(async () => {
    if (typeof document === "undefined") return;
    try {
      if (document.fullscreenElement) {
        await document.exitFullscreen();
      } else {
        await document.documentElement.requestFullscreen();
      }
    } catch {
      setPwaStatus("Fullscreen unavailable; Veles remains usable in this window.");
    }
  }, []);

  const requestPersistence = useCallback(async () => {
    try {
      const granted = (await navigator.storage?.persist?.()) ?? false;
      setPersisted(granted);
    } catch {
      // Ignore storage persistence errors on unsupported origins
    }
  }, []);

  useEffect(() => {
    destroyedRef.current = false;
    setWakeLockSupported(isWakeLockSupported());
    void navigator.storage
      ?.persisted?.()
      .then((isPersisted) => {
        if (isPersisted) {
          setPersisted(true);
          return;
        }
        return navigator.storage?.persist?.().then(setPersisted);
      })
      .catch(() => {});

    const standaloneQuery = window.matchMedia("(display-mode: standalone)");
    const onDisplayChange = () => setIsStandalone(standaloneQuery.matches);
    const onOnline = () => setNetworkOnline(navigator.onLine);
    const onBeforeInstallPrompt = (event: Event) => {
      event.preventDefault();
      setInstallPrompt(event as BeforeInstallPromptEvent);
      setPwaStatus("Install is available; opens Veles in standalone mode.");
    };
    const onAppInstalled = () => {
      setInstallPrompt(null);
      setPwaStatus("Veles is installed and its static app shell is cached.");
      onDisplayChange();
    };
    const onVisibilityChange = () => {
      setWakeLockSupported(isWakeLockSupported());
      if (
        document.visibilityState === "visible" &&
        wakeLockDesiredRef.current &&
        !wakeLockRef.current
      ) {
        void acquireWakeLock();
      }
    };
    const onFullscreenChange = () => {
      setFullscreenActive(Boolean(document.fullscreenElement));
    };
    const onSwUpdateReadyEvent = () => {
      setUpdateAvailable(true);
      setPwaStatus("An app update is ready. Apply it between runs.");
    };

    updateSwRef.current = registerSW({
      immediate: true,
      onOfflineReady: () => {
        if (!destroyedRef.current) {
          setPwaStatus(
            "Offline app shell is ready. Remote LLM and search calls still need a network.",
          );
        }
      },
      onNeedRefresh: () => {
        if (!destroyedRef.current) {
          setUpdateAvailable(true);
          setPwaStatus("An app update is ready. Apply it between runs.");
        }
      },
      onRegisteredSW: (_scriptUrl, registration) => {
        if (destroyedRef.current || !import.meta.env.PROD) return;
        setPwaStatus(resolveRegisteredSwMessage(registration));
      },
      onRegisterError: (error) => {
        if (!destroyedRef.current) {
          setPwaStatus(`Offline app shell failed: ${formatErrorMessage(error)}`);
        }
      },
    });

    standaloneQuery.addEventListener("change", onDisplayChange);
    window.addEventListener("online", onOnline);
    window.addEventListener("offline", onOnline);
    window.addEventListener("beforeinstallprompt", onBeforeInstallPrompt);
    window.addEventListener("appinstalled", onAppInstalled);
    window.addEventListener("veles:sw-update-ready", onSwUpdateReadyEvent);
    document.addEventListener("visibilitychange", onVisibilityChange);
    document.addEventListener("fullscreenchange", onFullscreenChange);

    return () => {
      destroyedRef.current = true;
      standaloneQuery.removeEventListener("change", onDisplayChange);
      window.removeEventListener("online", onOnline);
      window.removeEventListener("offline", onOnline);
      window.removeEventListener("beforeinstallprompt", onBeforeInstallPrompt);
      window.removeEventListener("appinstalled", onAppInstalled);
      window.removeEventListener("veles:sw-update-ready", onSwUpdateReadyEvent);
      document.removeEventListener("visibilitychange", onVisibilityChange);
      document.removeEventListener("fullscreenchange", onFullscreenChange);
      void releaseWakeLock();
    };
  }, [acquireWakeLock, releaseWakeLock]);

  const supported = wakeLockSupported || isWakeLockSupported();
  return {
    displayMode: isStandalone ? "installed app" : "browser tab",
    isStandalone,
    networkOnline,
    canInstall: !isStandalone && installPrompt !== null,
    installButtonLabel: computeInstallButtonLabel(
      isStandalone,
      installPrompt !== null,
    ),
    updateAvailable,
    applyingUpdate,
    pwaStatus,
    wakeLockSupported: supported,
    wakeLockActive,
    wakeLockPending,
    wakeLockButtonLabel: computeWakeLockButtonLabel(
      wakeLockActive,
      wakeLockPending,
      wakeLockDesired,
    ),
    wakeLockStatus,
    fullscreenActive,
    fullscreenButtonLabel: fullscreenActive
      ? "Exit app fullscreen"
      : "Fullscreen app",
    persisted,
    triggerInstall,
    triggerUpdate,
    toggleWakeLock,
    toggleAppFullscreen,
    requestPersistence,
  };
}
