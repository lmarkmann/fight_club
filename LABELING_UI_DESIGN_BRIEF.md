# Fight Club - Labeling UI Design Brief

> **Document Purpose**: Instructions for a frontend designer to create a mockup for a boxing action labeling and data collection interface.

---

## 1. Project Context

### What We're Building
A **video annotation tool** for labeling boxing actions in fight videos. The labeled data will train a machine learning model to automatically recognize 13 types of boxing punches and movements.

### Why This Matters
- Each video frame contains pose/skeleton data (stick figure representation of the fighter)
- Human labelers need to watch video segments and tag what action is being performed
- The UI must make this process **fast, accurate, and ergonomic** for labeling thousands of video clips

---

## 2. The 13 Action Classes to Label

The labeler will assign ONE of these 13 actions to each video segment:

| ID | Action Name | Hotkey | Category |
|----|-------------|--------|----------|
| 0 | Jab to Head | `1` | Straight |
| 1 | Jab to Body | `2` | Straight |
| 2 | Cross to Head | `3` | Straight |
| 3 | Cross to Body | `4` | Straight |
| 4 | Lead Hook to Head | `5` | Hook |
| 5 | Lead Hook to Body | `6` | Hook |
| 6 | Rear Hook to Head | `7` | Hook |
| 7 | Rear Hook to Body | `8` | Hook |
| 8 | Lead Uppercut | `9` | Uppercut |
| 9 | Rear Uppercut | `0` | Uppercut |
| 10 | Overhand | `O` | Other |
| 11 | Defensive Movement | `D` | Other |
| 12 | Idle / Stance | `I` | Other |

### Design Requirements for Action Selection
- Display all 13 actions in a **visible panel** (not hidden in dropdowns)
- Group actions by category (Straights, Hooks, Uppercuts, Other) with visual separation
- Show the **hotkey** prominently next to each action button
- Highlight the currently selected action
- Use **color coding** by category:
  - Straights: Blue tones
  - Hooks: Green tones
  - Uppercuts: Orange tones
  - Other: Gray tones

---

## 3. Core UI Components

### 3.1 Video Player Area

**Requirements:**
- Large, central video display (minimum 60% of viewport width)
- Frame-by-frame navigation controls:
  - Play / Pause button
  - Step forward 1 frame (`→` arrow key)
  - Step backward 1 frame (`←` arrow key)
  - Jump forward 1 second (`Shift + →`)
  - Jump backward 1 second (`Shift + ←`)
- Playback speed control (0.25x, 0.5x, 1x, 2x)
- Current timestamp display in format: `MM:SS.mmm` (minutes, seconds, milliseconds)
- Current frame number display: `Frame 1234 / 5000`
- Scrubber/timeline bar for quick navigation

**Pose Overlay Toggle:**
- Checkbox or toggle to show/hide skeleton overlay on video
- When enabled: draw stick figure (17 keypoints connected by lines) over the fighter
- Skeleton should use bright color (e.g., cyan/lime) for visibility

### 3.2 Timeline / Segment Editor

**Requirements:**
- Horizontal timeline showing the full video duration
- Visual representation of labeled segments as colored blocks on the timeline
- Each segment block shows:
  - Color matching its action category
  - Action name abbreviation (e.g., "JAB-H", "CROSS-B", "L-HOOK-H")
- Ability to:
  - Set segment **start point** (`[` key or "Mark In" button)
  - Set segment **end point** (`]` key or "Mark Out" button)
  - Current selection highlighted with handles for adjustment
- Segment minimum duration indicator (minimum 5 frames required)
- Zoom in/out on timeline for precise editing

### 3.3 Action Selection Panel

**Requirements:**
- Always visible (not collapsible) during labeling
- Layout: 4 rows grouped by category

```
┌─────────────────────────────────────────────────┐
│ STRAIGHTS                                       │
│ [1] Jab Head   [2] Jab Body                     │
│ [3] Cross Head [4] Cross Body                   │
├─────────────────────────────────────────────────┤
│ HOOKS                                           │
│ [5] Lead Hook Head  [6] Lead Hook Body          │
│ [7] Rear Hook Head  [8] Rear Hook Body          │
├─────────────────────────────────────────────────┤
│ UPPERCUTS                                       │
│ [9] Lead Uppercut   [0] Rear Uppercut           │
├─────────────────────────────────────────────────┤
│ OTHER                                           │
│ [O] Overhand  [D] Defensive  [I] Idle/Stance    │
└─────────────────────────────────────────────────┘
```

- Clicking a button OR pressing its hotkey assigns that action to the current segment
- Visual feedback: button pulses/highlights when pressed
- After assigning action, auto-advance to next unlabeled region (optional toggle)

### 3.4 Segment List Panel

**Requirements:**
- Scrollable list of all labeled segments for current video
- Each row displays:
  - Segment number (#1, #2, #3...)
  - Start time → End time
  - Action name with category color indicator
  - Duration in seconds
  - Delete button (trash icon)
  - Edit button (pencil icon) - clicking jumps to that segment
- Sort options: by time, by action type
- Filter options: show only specific action categories
- Total count summary at bottom: "47 segments labeled"

### 3.5 Video Queue / File Browser

**Requirements:**
- Sidebar or modal showing list of videos to label
- Each video entry shows:
  - Thumbnail (first frame)
  - Filename
  - Duration
  - Resolution
  - Labeling status: "Not started" / "In progress (12/~30)" / "Complete"
  - Quality indicator (green check if meets requirements, yellow warning if borderline)
- Ability to load next video in queue
- Progress bar showing overall labeling progress across all videos

---

## 4. Video Quality Requirements Display

When a video is loaded, show a **quality status panel**:

| Metric | Minimum Required | Status |
|--------|------------------|--------|
| FPS | 24.0 | ✓ Pass / ✗ Fail |
| Resolution | 640 × 480 | ✓ Pass / ✗ Fail |
| Bitrate | 1000 kbps | ✓ Pass / ✗ Fail |
| Duration | 30 seconds | ✓ Pass / ✗ Fail |

- If video fails requirements, show warning banner but still allow labeling
- Display detected codec (h264, hevc, etc.)

---

## 5. Keyboard Shortcuts Reference

Design a **help overlay** (triggered by `?` key) showing all shortcuts:

```
PLAYBACK
  Space       Play / Pause
  ←           Previous frame
  →           Next frame
  Shift+←     Back 1 second
  Shift+→     Forward 1 second

SEGMENT EDITING
  [           Mark segment start (In point)
  ]           Mark segment end (Out point)
  Enter       Confirm segment with selected action
  Backspace   Delete current segment
  Escape      Cancel current selection

ACTION HOTKEYS
  1-9, 0      Select action (see panel)
  O           Overhand
  D           Defensive Movement
  I           Idle/Stance

NAVIGATION
  N           Next video in queue
  P           Previous video in queue
  ?           Show/hide this help
```

---

## 6. Labeling Workflow (User Journey)

Design the UI to support this workflow:

```
1. LOAD VIDEO
   └── User selects video from queue
   └── Video loads, quality check runs
   └── Pose extraction happens (show progress bar)

2. NAVIGATE TO ACTION
   └── User scrubs/plays video to find a punch
   └── Pose skeleton overlay helps identify movement

3. MARK SEGMENT
   └── User presses [ at start of punch
   └── User presses ] at end of punch
   └── Timeline shows highlighted selection

4. ASSIGN ACTION
   └── User presses hotkey (e.g., "3" for Cross to Head)
   └── Segment is saved and colored on timeline
   └── Optional: auto-advance to next unlabeled area

5. REPEAT
   └── Continue until video fully labeled

6. NEXT VIDEO
   └── Mark video as complete
   └── Load next video from queue
```

---

## 7. Data Export Preview (Read-Only Display)

Show a preview of what will be exported (labeler doesn't edit this directly):

```json
{
  "video_file": "fight_001.mp4",
  "labeled_by": "annotator_username",
  "date": "2026-01-30",
  "segments": [
    {
      "id": 1,
      "action": "CROSS_HEAD",
      "action_id": 2,
      "start_time": 12.450,
      "end_time": 13.120,
      "start_frame": 299,
      "end_frame": 315
    }
  ]
}
```

This helps labelers understand that their work is being captured correctly.

---

## 8. Visual Design Guidelines

### Color Palette

| Element | Color | Hex |
|---------|-------|-----|
| Straights category | Blue | `#3B82F6` |
| Hooks category | Green | `#22C55E` |
| Uppercuts category | Orange | `#F97316` |
| Other category | Gray | `#6B7280` |
| Pose skeleton | Cyan | `#06B6D4` |
| Selection highlight | Yellow | `#FACC15` |
| Background | Dark gray | `#1F2937` |
| Panel background | Darker gray | `#111827` |
| Text primary | White | `#F9FAFB` |
| Text secondary | Light gray | `#9CA3AF` |

### Typography
- Use monospace font for timestamps and frame numbers
- Use sans-serif for labels and UI text
- Hotkeys should be displayed in a "keyboard key" style badge

### Layout Recommendation

```
┌────────────────────────────────────────────────────────────────┐
│  HEADER: Video name | Progress | Quality status | Save button  │
├──────────────────────────────────┬─────────────────────────────┤
│                                  │                             │
│                                  │     ACTION SELECTION        │
│       VIDEO PLAYER               │        PANEL                │
│       (with pose overlay)        │                             │
│                                  │     [Grouped buttons        │
│                                  │      with hotkeys]          │
│                                  │                             │
├──────────────────────────────────┼─────────────────────────────┤
│                                  │                             │
│       TIMELINE / SCRUBBER        │     SEGMENT LIST            │
│       [colored segment blocks]   │     (scrollable)            │
│                                  │                             │
├──────────────────────────────────┴─────────────────────────────┤
│  FOOTER: Keyboard shortcuts hint | Total segments | Auto-save  │
└────────────────────────────────────────────────────────────────┘
```

---

## 9. States to Design

### Empty States
- No video loaded: "Select a video from the queue to begin labeling"
- No segments yet: "Mark your first segment using [ and ] keys"
- Queue empty: "All videos have been labeled!"

### Loading States
- Video loading: Progress bar with "Loading video..."
- Pose extraction: Progress bar with "Extracting pose data... (frame 234/5000)"
- Saving: Brief "Saved" toast notification

### Error States
- Video failed to load: "Could not load video. File may be corrupted."
- Video below quality threshold: Warning banner (yellow) with details
- Segment too short: "Segment must be at least 5 frames"

### Success States
- Segment saved: Brief highlight animation on timeline
- Video complete: Celebration indicator + prompt to load next
- All done: Summary statistics of labeling session

---

## 10. Responsive Considerations

- **Minimum viewport**: 1280 × 720 (this is a desktop power-user tool)
- **Optimal viewport**: 1920 × 1080
- Video player should scale while maintaining aspect ratio
- Panels can be resizable (drag borders)
- Support for dual-monitor setup (video on one, controls on other) is a nice-to-have

---

## 11. Accessibility Notes

- All interactive elements must be keyboard accessible
- Hotkeys should not conflict with screen reader shortcuts
- Sufficient color contrast (WCAG AA minimum)
- Focus indicators on all interactive elements
- Aria labels for icon-only buttons

---

## 12. Out of Scope for Mockup

The following are NOT needed in the mockup (backend/data concerns):

- Actual video file handling
- Real pose extraction processing
- Database or file storage
- User authentication
- Multi-user collaboration features
- Actual export functionality

Focus purely on the **visual design and interaction patterns**.

---

## 13. Deliverables Checklist

Please provide mockups for:

- [ ] Main labeling interface (all components visible)
- [ ] Video loading state
- [ ] Empty state (no segments)
- [ ] Active labeling state (segment being marked)
- [ ] Completed video state
- [ ] Video queue/browser view
- [ ] Keyboard shortcuts help overlay
- [ ] Mobile/tablet "not supported" message

---

## 14. Reference: Pose Skeleton Visualization

The skeleton overlay connects these 17 body points:

```
           0 (nose)
          /   \
     1 (L eye) 2 (R eye)
        |       |
     3 (L ear) 4 (R ear)

     5 -------- 6
   (L shoulder) (R shoulder)
     |          |
     7          8
   (L elbow)   (R elbow)
     |          |
     9          10
   (L wrist)   (R wrist)

    11 -------- 12
   (L hip)     (R hip)
     |          |
    13          14
   (L knee)    (R knee)
     |          |
    15          16
   (L ankle)   (R ankle)
```

For boxing, the **upper body (points 0-12)** is most important.

---

## 15. Questions for Designer

1. Do you have a preferred design tool (Figma, Sketch, Adobe XD)?
2. Should we include a dark mode AND light mode, or dark only?
3. Any existing design system or component library to align with?
4. Timeline for first mockup iteration?

---

*Document created: 2026-01-30*
*Project: Fight Club - Boxing Action Classification*
*Contact: [Add project owner contact]*
