import { createContext, useCallback, useContext, useEffect, useId, useRef, useState } from "react";
import type { ReactNode } from "react";
import { createPortal } from "react-dom";

type Notice = { id: number; source: string; message: string; title?: string; error: boolean };
type Notify = (source: string, message: string | null, error: boolean, title?: string) => void;
const NotificationContext = createContext<Notify | null>(null);

export function NotificationProvider({ children }: { children: ReactNode }) {
  const [notices, setNotices] = useState<Notice[]>([]);
  const nextId = useRef(0);
  const notify = useCallback<Notify>((source, message, error, title) => {
    const notice = message ? { id: ++nextId.current, source, message, error, title } : null;
    setNotices((current) => {
      const others = current.filter((item) => item.source !== source);
      return notice ? [...others, notice].slice(-4) : others;
    });
  }, []);
  const dismiss = useCallback((id: number) => setNotices((current) => current.filter((item) => item.id !== id)), []);
  return <NotificationContext.Provider value={notify}>
    {children}
    {createPortal(<div className="notification-stack" aria-label="Notifications">
      {notices.map((notice) => <Notification key={notice.id} notice={notice} dismiss={dismiss} />)}
    </div>, document.body)}
  </NotificationContext.Provider>;
}

function Notification({ notice, dismiss }: { notice: Notice; dismiss: (id: number) => void }) {
  const [hovered, setHovered] = useState(false);
  const [focused, setFocused] = useState(false);
  useEffect(() => {
    if (hovered || focused) return;
    const timer = window.setTimeout(() => dismiss(notice.id), notice.error ? 10000 : 6000);
    return () => window.clearTimeout(timer);
  }, [dismiss, notice.id, notice.error, hovered, focused]);
  return <div className={`notification${notice.error ? " notification-error" : ""}`}
    role={notice.error ? "alert" : "status"} aria-atomic="true"
    onMouseEnter={() => setHovered(true)} onMouseLeave={() => setHovered(false)}
    onFocus={() => setFocused(true)} onBlur={(event) => {
      if (!event.currentTarget.contains(event.relatedTarget)) setFocused(false);
    }}>
    <div>{notice.title && <strong>{notice.title}</strong>}<span>{notice.message}</span></div>
    <button className="icon-button" aria-label="Dismiss notification" onClick={() => dismiss(notice.id)}>×</button>
  </div>;
}

export function useNotification(error = false, title?: string) {
  const notify = useContext(NotificationContext);
  const source = useId();
  if (!notify) throw new Error("Notifications require NotificationProvider.");
  return useCallback((message: string | null, isError = error) => notify(source, message, isError, title), [notify, source, error, title]);
}
