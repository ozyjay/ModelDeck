import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { NotificationProvider, useNotification } from "./Notifications";

function Actions() {
  const notify = useNotification();
  const fail = useNotification(true, "Action failed");
  return <main><button onClick={() => notify("Rehearsal passed")}>Rehearse</button>
    <button onClick={() => fail("Worker unavailable")}>Fail</button></main>;
}

describe("temporary notifications", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => { cleanup(); vi.useRealTimers(); });

  it("renders outside the page layout, expires, and repeats the same message", () => {
    const { container } = render(<NotificationProvider><Actions /></NotificationProvider>);
    fireEvent.click(screen.getByText("Rehearse"));
    expect(screen.getByRole("status")).toHaveTextContent("Rehearsal passed");
    expect(container.querySelector(".notification")).toBeNull();
    act(() => vi.advanceTimersByTime(5000));
    fireEvent.click(screen.getByText("Rehearse"));
    act(() => vi.advanceTimersByTime(5000));
    expect(screen.getByRole("status")).toBeInTheDocument();
    act(() => vi.advanceTimersByTime(1000));
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  it("pauses expiry while hovered or focused and supports manual dismissal", () => {
    render(<NotificationProvider><Actions /></NotificationProvider>);
    fireEvent.click(screen.getByText("Rehearse"));
    const notice = screen.getByRole("status");
    fireEvent.mouseEnter(notice);
    act(() => vi.advanceTimersByTime(20000));
    expect(notice).toBeInTheDocument();
    const dismiss = screen.getByRole("button", { name: "Dismiss notification" });
    fireEvent.focus(dismiss);
    fireEvent.mouseLeave(notice);
    act(() => vi.advanceTimersByTime(20000));
    expect(notice).toBeInTheDocument();
    fireEvent.click(dismiss);
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  it("stacks independent feedback and gives errors longer to be read", () => {
    render(<NotificationProvider><Actions /></NotificationProvider>);
    fireEvent.click(screen.getByText("Rehearse"));
    fireEvent.click(screen.getByText("Fail"));
    expect(screen.getByRole("alert")).toHaveTextContent("Worker unavailable");
    act(() => vi.advanceTimersByTime(6000));
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
    expect(screen.getByRole("alert")).toBeInTheDocument();
    act(() => vi.advanceTimersByTime(4000));
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });
});
