# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========

import datetime
import io
import json
import os
import random
import re
import shutil
import time
import urllib.parse
from copy import deepcopy
from typing import (
    TYPE_CHECKING,
    Any,
    BinaryIO,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    TypedDict,
    Union,
    cast,
)

from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
    from camel.agents import ChatAgent
from camel.logger import get_logger
from camel.messages import BaseMessage
from camel.models import BaseModelBackend, ModelFactory
from camel.toolkits import FunctionTool, VideoAnalysisToolkit
from camel.toolkits.base import BaseToolkit
from camel.types import ModelPlatformType, ModelType
from camel.utils import dependencies_required, retry_on_error

logger = get_logger(__name__)

TOP_NO_LABEL_ZONE = 20

AVAILABLE_ACTIONS_PROMPT = """
1. `fill_input_id(identifier: Union[str, int], text: str)`: 在指定的输入框中填入文本并按回车键。
2. `click_id(identifier: Union[str, int])`: 点击具有指定ID的元素。
3. `hover_id(identifier: Union[str, int])`: 将鼠标悬停在具有指定ID的元素上。
4. `download_file_id(identifier: Union[str, int])`: 下载具有指定ID的文件。返回下载文件的路径。如果文件下载成功,您可以停止模拟并报告下载文件的路径以供进一步处理。
5. `scroll_to_bottom()`: 滚动到页面底部。
6. `scroll_to_top()`: 滚动到页面顶部。
7. `scroll_up()`: 向上滚动页面。当您想查看当前视口上方的元素时使用。
8. `scroll_down()`: 向下滚动页面。当您想查看当前视口下方的元素时使用。如果网页没有变化,说明已经滚动到底部。
9. `back()`: 返回上一页。当当前页面无用时,这个功能很有用。
10. `stop()`: 停止操作过程,因为任务已完成或失败(无法找到答案)。在这种情况下,您应该在输出中提供您的答案。
11. `get_url()`: 获取当前页面的URL。
12. `find_text_on_page(search_text: str)`: 在当前整个页面中查找指定文本,并将页面滚动到目标文本处。这相当于按Ctrl + F并搜索文本,当您想快速检查当前页面是否包含某些特定文本时非常有用。
13. `visit_page(url: str)`: 访问指定的URL页面。
14. `click_blank_area()`: 点击页面的空白区域以取消当前元素的焦点。当您点击了一个元素但它无法自动取消焦点(例如菜单栏)以自动渲染更新的网页时,这很有用。
15. `ask_question_about_video(question: str)`: 询问关于当前包含视频的网页的问题,例如YouTube网站。
"""

ACTION_WITH_FEEDBACK_LIST = [
    'ask_question_about_video',
    'download_file_id',
    'find_text_on_page',
]


# Code from magentic-one
class DOMRectangle(TypedDict):
    x: Union[int, float]
    y: Union[int, float]
    width: Union[int, float]
    height: Union[int, float]
    top: Union[int, float]
    right: Union[int, float]
    bottom: Union[int, float]
    left: Union[int, float]


class VisualViewport(TypedDict):
    height: Union[int, float]
    width: Union[int, float]
    offsetLeft: Union[int, float]
    offsetTop: Union[int, float]
    pageLeft: Union[int, float]
    pageTop: Union[int, float]
    scale: Union[int, float]
    clientWidth: Union[int, float]
    clientHeight: Union[int, float]
    scrollWidth: Union[int, float]
    scrollHeight: Union[int, float]


class InteractiveRegion(TypedDict):
    tag_name: str
    role: str
    aria_name: str
    v_scrollable: bool
    rects: List[DOMRectangle]


def _get_str(d: Any, k: str) -> str:
    r"""Safely retrieve a string value from a dictionary."""
    if k not in d:
        raise KeyError(f"Missing required key: '{k}'")
    val = d[k]
    if isinstance(val, str):
        return val
    raise TypeError(
        f"Expected a string for key '{k}', " f"but got {type(val).__name__}"
    )


def _get_number(d: Any, k: str) -> Union[int, float]:
    r"""Safely retrieve a number (int or float) from a dictionary"""
    val = d[k]
    if isinstance(val, (int, float)):
        return val
    raise TypeError(
        f"Expected a number (int/float) for key "
        f"'{k}', but got {type(val).__name__}"
    )


def _get_bool(d: Any, k: str) -> bool:
    r"""Safely retrieve a boolean value from a dictionary."""
    val = d[k]
    if isinstance(val, bool):
        return val
    raise TypeError(
        f"Expected a boolean for key '{k}', " f"but got {type(val).__name__}"
    )


def _parse_json_output(text: str) -> Dict[str, Any]:
    r"""Extract JSON output from a string."""

    markdown_pattern = r'```(?:json)?\s*(.*?)\s*```'
    markdown_match = re.search(markdown_pattern, text, re.DOTALL)
    if markdown_match:
        text = markdown_match.group(1).strip()

    triple_quotes_pattern = r'"""(?:json)?\s*(.*?)\s*"""'
    triple_quotes_match = re.search(triple_quotes_pattern, text, re.DOTALL)
    if triple_quotes_match:
        text = triple_quotes_match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            fixed_text = re.sub(
                r'`([^`]*?)`(?=\s*[:,\[\]{}]|$)', r'"\1"', text
            )
            return json.loads(fixed_text)
        except json.JSONDecodeError:
            result = {}
            try:
                bool_pattern = r'"(\w+)"\s*:\s*(true|false)'
                for match in re.finditer(bool_pattern, text, re.IGNORECASE):
                    key, value = match.groups()
                    result[key] = value.lower() == "true"

                str_pattern = r'"(\w+)"\s*:\s*"([^"]*)"'
                for match in re.finditer(str_pattern, text):
                    key, value = match.groups()
                    result[key] = value

                num_pattern = r'"(\w+)"\s*:\s*(-?\d+(?:\.\d+)?)'
                for match in re.finditer(num_pattern, text):
                    key, value = match.groups()
                    try:
                        result[key] = int(value)
                    except ValueError:
                        result[key] = float(value)

                empty_str_pattern = r'"(\w+)"\s*:\s*""'
                for match in re.finditer(empty_str_pattern, text):
                    key = match.group(1)
                    result[key] = ""

                if result:
                    return result

                logger.warning(f"Failed to parse JSON output: {text}")
                return {}
            except Exception as e:
                logger.warning(f"Error while extracting fields from JSON: {e}")
                return {}


def _reload_image(image: Image.Image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return Image.open(buffer)


def dom_rectangle_from_dict(rect: Dict[str, Any]) -> DOMRectangle:
    r"""Create a DOMRectangle object from a dictionary."""
    return DOMRectangle(
        x=_get_number(rect, "x"),
        y=_get_number(rect, "y"),
        width=_get_number(rect, "width"),
        height=_get_number(rect, "height"),
        top=_get_number(rect, "top"),
        right=_get_number(rect, "right"),
        bottom=_get_number(rect, "bottom"),
        left=_get_number(rect, "left"),
    )


def interactive_region_from_dict(region: Dict[str, Any]) -> InteractiveRegion:
    r"""Create an :class:`InteractiveRegion` object from a dictionary."""
    typed_rects: List[DOMRectangle] = []
    for rect in region["rects"]:
        typed_rects.append(dom_rectangle_from_dict(rect))

    return InteractiveRegion(
        tag_name=_get_str(region, "tag_name"),
        role=_get_str(region, "role"),
        aria_name=_get_str(region, "aria-name"),
        v_scrollable=_get_bool(region, "v-scrollable"),
        rects=typed_rects,
    )


def visual_viewport_from_dict(viewport: Dict[str, Any]) -> VisualViewport:
    r"""Create a :class:`VisualViewport` object from a dictionary."""
    return VisualViewport(
        height=_get_number(viewport, "height"),
        width=_get_number(viewport, "width"),
        offsetLeft=_get_number(viewport, "offsetLeft"),
        offsetTop=_get_number(viewport, "offsetTop"),
        pageLeft=_get_number(viewport, "pageLeft"),
        pageTop=_get_number(viewport, "pageTop"),
        scale=_get_number(viewport, "scale"),
        clientWidth=_get_number(viewport, "clientWidth"),
        clientHeight=_get_number(viewport, "clientHeight"),
        scrollWidth=_get_number(viewport, "scrollWidth"),
        scrollHeight=_get_number(viewport, "scrollHeight"),
    )


def add_set_of_mark(
    screenshot: Union[bytes, Image.Image, io.BufferedIOBase],
    ROIs: Dict[str, InteractiveRegion],
) -> Tuple[Image.Image, List[str], List[str], List[str]]:
    if isinstance(screenshot, Image.Image):
        return _add_set_of_mark(screenshot, ROIs)

    if isinstance(screenshot, bytes):
        screenshot = io.BytesIO(screenshot)

    image = Image.open(cast(BinaryIO, screenshot))
    comp, visible_rects, rects_above, rects_below = _add_set_of_mark(
        image, ROIs
    )
    image.close()
    return comp, visible_rects, rects_above, rects_below


def _add_set_of_mark(
    screenshot: Image.Image, ROIs: Dict[str, InteractiveRegion]
) -> Tuple[Image.Image, List[str], List[str], List[str]]:
    r"""Add a set of marks to the screenshot.

    Args:
        screenshot (Image.Image): The screenshot to add marks to.
        ROIs (Dict[str, InteractiveRegion]): The regions to add marks to.

    Returns:
        Tuple[Image.Image, List[str], List[str], List[str]]: A tuple
        containing the screenshot with marked ROIs, ROIs fully within the
        images, ROIs located above the visible area, and ROIs located below
        the visible area.
    """
    visible_rects: List[str] = list()
    rects_above: List[str] = list()  # Scroll up to see
    rects_below: List[str] = list()  # Scroll down to see

    fnt = ImageFont.load_default(14)
    base = screenshot.convert("L").convert("RGBA")
    overlay = Image.new("RGBA", base.size)

    draw = ImageDraw.Draw(overlay)
    for r in ROIs:
        for rect in ROIs[r]["rects"]:
            # Empty rectangles
            if not rect or rect["width"] == 0 or rect["height"] == 0:
                continue

            # TODO: add scroll left and right?
            horizontal_center = (rect["right"] + rect["left"]) / 2.0
            vertical_center = (rect["top"] + rect["bottom"]) / 2.0
            is_within_horizon = 0 <= horizontal_center < base.size[0]
            is_above_viewport = vertical_center < 0
            is_below_viewport = vertical_center >= base.size[1]

            if is_within_horizon:
                if is_above_viewport:
                    rects_above.append(r)
                elif is_below_viewport:
                    rects_below.append(r)
                else:  # Fully visible
                    visible_rects.append(r)
                    _draw_roi(draw, int(r), fnt, rect)

    comp = Image.alpha_composite(base, overlay)
    overlay.close()
    return comp, visible_rects, rects_above, rects_below


def _draw_roi(
    draw: ImageDraw.ImageDraw,
    idx: int,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    rect: DOMRectangle,
) -> None:
    r"""Draw a ROI on the image.

    Args:
        draw (ImageDraw.ImageDraw): The draw object.
        idx (int): The index of the ROI.
        font (ImageFont.FreeTypeFont | ImageFont.ImageFont): The font.
        rect (DOMRectangle): The DOM rectangle.
    """
    color = _get_random_color(idx)
    text_color = _get_text_color(color)

    roi = ((rect["left"], rect["top"]), (rect["right"], rect["bottom"]))

    label_location = (rect["right"], rect["top"])
    label_anchor = "rb"

    if label_location[1] <= TOP_NO_LABEL_ZONE:
        label_location = (rect["right"], rect["bottom"])
        label_anchor = "rt"

    draw.rectangle(
        roi, outline=color, fill=(color[0], color[1], color[2], 48), width=2
    )

    bbox = draw.textbbox(
        label_location,
        str(idx),
        font=font,
        anchor=label_anchor,
        align="center",
    )
    bbox = (bbox[0] - 3, bbox[1] - 3, bbox[2] + 3, bbox[3] + 3)
    draw.rectangle(bbox, fill=color)

    draw.text(
        label_location,
        str(idx),
        fill=text_color,
        font=font,
        anchor=label_anchor,
        align="center",
    )


def _get_text_color(
    bg_color: Tuple[int, int, int, int],
) -> Tuple[int, int, int, int]:
    r"""Determine the ideal text color (black or white) for contrast.

    Args:
        bg_color: The background color (R, G, B, A).

    Returns:
        A tuple representing black or white color for text.
    """
    luminance = bg_color[0] * 0.3 + bg_color[1] * 0.59 + bg_color[2] * 0.11
    return (0, 0, 0, 255) if luminance > 120 else (255, 255, 255, 255)


def _get_random_color(identifier: int) -> Tuple[int, int, int, int]:
    r"""Generate a consistent random RGBA color based on the identifier.

    Args:
        identifier: The ID used as a seed to ensure color consistency.

    Returns:
        A tuple representing (R, G, B, A) values.
    """
    rnd = random.Random(int(identifier))
    r = rnd.randint(0, 255)
    g = rnd.randint(125, 255)
    b = rnd.randint(0, 50)
    color = [r, g, b]
    # TODO: check why shuffle is needed?
    rnd.shuffle(color)
    color.append(255)
    return cast(Tuple[int, int, int, int], tuple(color))


class BaseBrowser:
    def __init__(
        self,
        headless=True,
        cache_dir: Optional[str] = None,
        channel: Literal["chrome", "msedge", "chromium"] = "chromium",
    ):
        r"""Initialize the WebBrowser instance.

        Args:
            headless (bool): Whether to run the browser in headless mode.
            cache_dir (Union[str, None]): The directory to store cache files.
            channel (Literal["chrome", "msedge", "chromium"]): The browser
                channel to use. Must be one of "chrome", "msedge", or
                "chromium".

        Returns:
            None
        """
        from playwright.sync_api import (
            sync_playwright,
        )

        self.history: list = []
        self.headless = headless
        self.channel = channel
        self._ensure_browser_installed()
        self.playwright = sync_playwright().start()
        self.page_history: list = []  # stores the history of visited pages

        # Set the cache directory
        self.cache_dir = "tmp/" if cache_dir is None else cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        # Load the page script
        abs_dir_path = os.path.dirname(os.path.abspath(__file__))
        page_script_path = os.path.join(abs_dir_path, "page_script.js")

        try:
            with open(page_script_path, "r", encoding='utf-8') as f:
                self.page_script = f.read()
            f.close()
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Page script file not found at path: {page_script_path}"
            )

    def init(self) -> None:
        r"""Initialize the browser."""
        # Launch the browser, if headless is False, the browser will display
        self.browser = self.playwright.chromium.launch(
            headless=self.headless, channel=self.channel
        )
        # Create a new context
        self.context = self.browser.new_context(accept_downloads=True)
        # Create a new page
        self.page = self.context.new_page()

    def clean_cache(self) -> None:
        r"""Delete the cache directory and its contents."""
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)

    def _wait_for_load(self, timeout: int = 20) -> None:
        r"""Wait for a certain amount of time for the page to load."""
        timeout_ms = timeout * 1000

        self.page.wait_for_load_state("load", timeout=timeout_ms)

        # TODO: check if this is needed
        time.sleep(2)

    def click_blank_area(self) -> None:
        r"""Click a blank area of the page to unfocus the current element."""
        self.page.mouse.click(0, 0)
        self._wait_for_load()

    def visit_page(self, url: str) -> None:
        r"""Visit a page with the given URL."""

        self.page.goto(url)
        self._wait_for_load()
        self.page_url = url

    def ask_question_about_video(self, question: str) -> str:
        r"""Ask a question about the video on the current page,
        such as YouTube video.

        Args:
            question (str): The question to ask.

        Returns:
            str: The answer to the question.
        """
        video_analyzer = VideoAnalysisToolkit()
        result = video_analyzer.ask_question_about_video(
            self.page_url, question
        )
        return result

    @retry_on_error()
    def get_screenshot(
        self, save_image: bool = False
    ) -> Tuple[Image.Image, Union[str, None]]:
        r"""Get a screenshot of the current page.

        Args:
            save_image (bool): Whether to save the image to the cache
                directory.

        Returns:
            Tuple[Image.Image, str]: A tuple containing the screenshot
            image and the path to the image file if saved, otherwise
            :obj:`None`.
        """

        image_data = self.page.screenshot(timeout=60000)
        image = Image.open(io.BytesIO(image_data))

        file_path = None
        if save_image:
            # Get url name to form a file name
            # Use urlparser for a safer extraction the url name
            parsed_url = urllib.parse.urlparse(self.page_url)
            url_name = os.path.basename(str(parsed_url.path)) or "index"

            for char in ['\\', '/', ':', '*', '?', '"', '<', '>', '|', '.']:
                url_name = url_name.replace(char, "_")

            # Get formatted time: mmddhhmmss
            timestamp = datetime.datetime.now().strftime("%m%d%H%M%S")
            file_path = os.path.join(
                self.cache_dir, f"{url_name}_{timestamp}.png"
            )
            with open(file_path, "wb") as f:
                image.save(f, "PNG")
            f.close()

        return image, file_path

    def capture_full_page_screenshots(
        self, scroll_ratio: float = 0.8
    ) -> List[str]:
        r"""Capture full page screenshots by scrolling the page with a buffer
        zone.

        Args:
            scroll_ratio (float): The ratio of viewport height to scroll each
                step. (default: :obj:`0.8`)

        Returns:
            List[str]: A list of paths to the screenshot files.
        """
        screenshots = []
        scroll_height = self.page.evaluate("document.body.scrollHeight")
        assert self.page.viewport_size is not None
        viewport_height = self.page.viewport_size["height"]
        current_scroll = 0
        screenshot_index = 1

        max_height = scroll_height - viewport_height
        scroll_step = int(viewport_height * scroll_ratio)

        last_height = 0

        while True:
            logger.debug(
                f"Current scroll: {current_scroll}, max_height: "
                f"{max_height}, step: {scroll_step}"
            )

            _, file_path = self.get_screenshot(save_image=True)
            screenshots.append(file_path)

            self.page.evaluate(f"window.scrollBy(0, {scroll_step})")
            # Allow time for content to load
            time.sleep(0.5)

            current_scroll = self.page.evaluate("window.scrollY")
            # Break if there is no significant scroll
            if abs(current_scroll - last_height) < viewport_height * 0.1:
                break

            last_height = current_scroll
            screenshot_index += 1

        return screenshots

    def get_visual_viewport(self) -> VisualViewport:
        r"""Get the visual viewport of the current page.

        Returns:
            VisualViewport: The visual viewport of the current page.
        """
        try:
            self.page.evaluate(self.page_script)
        except Exception as e:
            logger.warning(f"Error evaluating page script: {e}")

        return visual_viewport_from_dict(
            self.page.evaluate("MultimodalWebSurfer.getVisualViewport();")
        )

    def get_interactive_elements(self) -> Dict[str, InteractiveRegion]:
        r"""Get the interactive elements of the current page.

        Returns:
            Dict[str, InteractiveRegion]: A dictionary of interactive elements.
        """
        try:
            self.page.evaluate(self.page_script)
        except Exception as e:
            logger.warning(f"Error evaluating page script: {e}")

        result = cast(
            Dict[str, Dict[str, Any]],
            self.page.evaluate("MultimodalWebSurfer.getInteractiveRects();"),
        )

        typed_results: Dict[str, InteractiveRegion] = {}
        for k in result:
            typed_results[k] = interactive_region_from_dict(result[k])

        return typed_results  # type: ignore[return-value]

    def get_som_screenshot(
        self,
        save_image: bool = False,
    ) -> Tuple[Image.Image, Union[str, None]]:
        r"""Get a screenshot of the current viewport with interactive elements
        marked.

        Args:
            save_image (bool): Whether to save the image to the cache
                directory.

        Returns:
            Tuple[Image.Image, str]: A tuple containing the screenshot image
                and the path to the image file.
        """

        self._wait_for_load()
        screenshot, _ = self.get_screenshot(save_image=False)
        rects = self.get_interactive_elements()

        file_path = None
        comp, visible_rects, rects_above, rects_below = add_set_of_mark(
            screenshot,
            rects,  # type: ignore[arg-type]
        )
        if save_image:
            parsed_url = urllib.parse.urlparse(self.page_url)
            url_name = os.path.basename(str(parsed_url.path)) or "index"
            for char in ['\\', '/', ':', '*', '?', '"', '<', '>', '|', '.']:
                url_name = url_name.replace(char, "_")
            timestamp = datetime.datetime.now().strftime("%m%d%H%M%S")
            file_path = os.path.join(
                self.cache_dir, f"{url_name}_{timestamp}.png"
            )
            with open(file_path, "wb") as f:
                comp.save(f, "PNG")
            f.close()

        return comp, file_path, rects

    def scroll_up(self) -> None:
        r"""Scroll up the page."""
        self.page.keyboard.press("PageUp")

    def scroll_down(self) -> None:
        r"""Scroll down the page."""
        self.page.keyboard.press("PageDown")

    def get_url(self) -> str:
        r"""Get the URL of the current page."""
        return self.page.url

    def click_id(self, identifier: Union[str, int]) -> None:
        r"""Click an element with the given identifier."""
        if isinstance(identifier, int):
            identifier = str(identifier)
        target = self.page.locator(f"[__elementId='{identifier}']")

        try:
            target.wait_for(timeout=5000)
        except (TimeoutError, Exception) as e:
            logger.debug(f"Error during click operation: {e}")
            raise ValueError("No such element.") from None

        target.scroll_into_view_if_needed()

        new_page = None
        try:
            with self.page.expect_event("popup", timeout=1000) as page_info:
                box = cast(Dict[str, Union[int, float]], target.bounding_box())
                self.page.mouse.click(
                    box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
                )
                new_page = page_info.value

                # If a new page is opened, switch to it
                if new_page:
                    self.page_history.append(deepcopy(self.page.url))
                    self.page = new_page

        except (TimeoutError, Exception) as e:
            logger.debug(f"Error during click operation: {e}")
            pass

        self._wait_for_load()

    def extract_url_content(self) -> str:
        r"""Extract the content of the current page."""
        content = self.page.content()
        return content

    def download_file_id(self, identifier: Union[str, int]) -> str:
        r"""Download a file with the given selector.

        Args:
            identifier (str): The identifier of the file to download.
            file_path (str): The path to save the downloaded file.

        Returns:
            str: The result of the action.
        """

        if isinstance(identifier, int):
            identifier = str(identifier)
        try:
            target = self.page.locator(f"[__elementId='{identifier}']")
        except (TimeoutError, Exception) as e:
            logger.debug(f"Error during download operation: {e}")
            logger.warning(
                f"Element with identifier '{identifier}' not found."
            )
            return f"Element with identifier '{identifier}' not found."

        target.scroll_into_view_if_needed()

        file_path = os.path.join(self.cache_dir)
        self._wait_for_load()

        try:
            with self.page.expect_download() as download_info:
                target.click()
                download = download_info.value
                file_name = download.suggested_filename

                file_path = os.path.join(file_path, file_name)
                download.save_as(file_path)

            return f"Downloaded file to path '{file_path}'."

        except (TimeoutError, Exception) as e:
            logger.debug(f"Error during download operation: {e}")
            return f"Failed to download file with identifier '{identifier}'."

    def fill_input_id(self, identifier: Union[str, int], text: str) -> str:
        r"""Fill an input field with the given text, and then press Enter.

        Args:
            identifier (str): The identifier of the input field.
            text (str): The text to fill.

        Returns:
            str: The result of the action.
        """
        if isinstance(identifier, int):
            identifier = str(identifier)

        try:
            target = self.page.locator(f"[__elementId='{identifier}']")
        except (TimeoutError, Exception) as e:
            logger.debug(f"Error during fill operation: {e}")
            logger.warning(
                f"Element with identifier '{identifier}' not found."
            )
            return f"Element with identifier '{identifier}' not found."

        target.scroll_into_view_if_needed()
        target.focus()
        try:
            target.fill(text)
        except (TimeoutError, Exception) as e:
            logger.debug(f"Error during fill operation: {e}")
            target.press_sequentially(text)

        target.press("Enter")
        self._wait_for_load()
        return (
            f"Filled input field '{identifier}' with text '{text}' "
            f"and pressed Enter."
        )

    def scroll_to_bottom(self) -> str:
        self.page.evaluate("window.scrollTo(0, document.body.scrollHeight);")
        self._wait_for_load()
        return "Scrolled to the bottom of the page."

    def scroll_to_top(self) -> str:
        self.page.evaluate("window.scrollTo(0, 0);")
        self._wait_for_load()
        return "Scrolled to the top of the page."

    def hover_id(self, identifier: Union[str, int]) -> str:
        r"""Hover over an element with the given identifier.

        Args:
            identifier (str): The identifier of the element to hover over.

        Returns:
            str: The result of the action.
        """
        if isinstance(identifier, int):
            identifier = str(identifier)
        try:
            target = self.page.locator(f"[__elementId='{identifier}']")
        except (TimeoutError, Exception) as e:
            logger.debug(f"Error during hover operation: {e}")
            logger.warning(
                f"Element with identifier '{identifier}' not found."
            )
            return f"Element with identifier '{identifier}' not found."

        target.scroll_into_view_if_needed()
        target.hover()
        self._wait_for_load()
        return f"Hovered over element with identifier '{identifier}'."

    def find_text_on_page(self, search_text: str) -> str:
        r"""Find the next given text on the page, and scroll the page to the
        targeted text. It is equivalent to pressing Ctrl + F and searching for
        the text.
        """
        # ruff: noqa: E501
        script = f"""
        (function() {{
            let text = "{search_text}";
            let found = window.find(text);
            if (!found) {{
                let elements = document.querySelectorAll("*:not(script):not(style)"); 
                for (let el of elements) {{
                    if (el.innerText && el.innerText.includes(text)) {{
                        el.scrollIntoView({{behavior: "smooth", block: "center"}});
                        el.style.backgroundColor = "yellow";
                        el.style.border = '2px solid red';
                        return true;
                    }}
                }}
                return false;
            }}
            return true;
        }})();
        """
        found = self.page.evaluate(script)
        self._wait_for_load()
        if found:
            return f"Found text '{search_text}' on the page."
        else:
            return f"Text '{search_text}' not found on the page."

    def back(self):
        r"""Navigate back to the previous page."""

        page_url_before = self.page.url
        self.page.go_back()

        page_url_after = self.page.url

        if page_url_after == "about:blank":
            self.visit_page(page_url_before)

        if page_url_before == page_url_after:
            # If the page is not changed, try to use the history
            if len(self.page_history) > 0:
                self.visit_page(self.page_history.pop())

        time.sleep(1)
        self._wait_for_load()

    def close(self):
        self.browser.close()

    # ruff: noqa: E501
    def show_interactive_elements(self):
        r"""Show simple interactive elements on the current page."""
        self.page.evaluate(self.page_script)
        self.page.evaluate("""
        () => {
            document.querySelectorAll('a, button, input, select, textarea, [tabindex]:not([tabindex="-1"]), [contenteditable="true"]').forEach(el => {
                el.style.border = '2px solid red';
            });
            }
        """)

    @retry_on_error()
    def get_webpage_content(self) -> str:
        from html2text import html2text

        self._wait_for_load()
        html_content = self.page.content()

        markdown_content = html2text(html_content)
        return markdown_content

    def _ensure_browser_installed(self) -> None:
        r"""Ensure the browser is installed."""
        import platform
        import subprocess
        import sys

        try:
            from playwright.sync_api import sync_playwright

            with sync_playwright() as p:
                browser = p.chromium.launch(channel=self.channel)
                browser.close()
        except Exception:
            logger.info("Installing Chromium browser...")
            try:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "playwright",
                        "install",
                        self.channel,
                    ],
                    check=True,
                    capture_output=True,
                )
                if platform.system().lower() == "linux":
                    subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "playwright",
                            "install-deps",
                            self.channel,
                        ],
                        check=True,
                        capture_output=True,
                    )
                logger.info("Chromium browser installation completed")
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to install browser: {e.stderr}")


class BrowserToolkit(BaseToolkit):
    r"""A class for browsing the web and interacting with web pages.

    This class provides methods for browsing the web and interacting with web
    pages.
    """

    def __init__(
        self,
        headless: bool = False,
        cache_dir: Optional[str] = None,
        channel: Literal["chrome", "msedge", "chromium"] = "chromium",
        history_window: int = 5,
        web_agent_model: Optional[BaseModelBackend] = None,
        planning_agent_model: Optional[BaseModelBackend] = None,
        output_language: str = "en",
    ):
        r"""Initialize the BrowserToolkit instance.

        Args:
            headless (bool): Whether to run the browser in headless mode.
            cache_dir (Union[str, None]): The directory to store cache files.
            channel (Literal["chrome", "msedge", "chromium"]): The browser
                channel to use. Must be one of "chrome", "msedge", or
                "chromium".
            history_window (int): The window size for storing the history of
                actions.
            web_agent_model (Optional[BaseModelBackend]): The model backend
                for the web agent.
            planning_agent_model (Optional[BaseModelBackend]): The model
                backend for the planning agent.
            output_language (str): The language to use for output.
                (default: :obj:`"en`")
        """

        self.browser = BaseBrowser(
            headless=headless, cache_dir=cache_dir, channel=channel
        )
        # This needs to be called explicitly
        # self.browser.init()

        self.history_window = history_window
        self.web_agent_model = web_agent_model
        self.planning_agent_model = planning_agent_model
        self.output_language = output_language

        self.history: list = []
        self.web_agent, self.planning_agent = self._initialize_agent()

    def _reset(self):
        self.web_agent.reset()
        self.planning_agent.reset()
        self.history = []
        os.makedirs(self.browser.cache_dir, exist_ok=True)

    def _initialize_agent(self) -> Tuple["ChatAgent", "ChatAgent"]:
        r"""Initialize the agent."""
        from camel.agents import ChatAgent

        if self.web_agent_model is None:
            web_agent_model = ModelFactory.create(
                model_platform=ModelPlatformType.OPENAI,
                model_type=ModelType.GPT_4O,
                model_config_dict={"temperature": 0, "top_p": 1},
            )
        else:
            web_agent_model = self.web_agent_model

        if self.planning_agent_model is None:
            planning_model = ModelFactory.create(
                model_platform=ModelPlatformType.OPENAI,
                model_type=ModelType.O3_MINI,
            )
        else:
            planning_model = self.planning_agent_model

        system_prompt = """
您是一个称职的网页代理,可以协助用户浏览网页。
给定一个高级任务,您可以利用预定义的浏览器工具来帮助用户实现他们的目标。
        """

        web_agent = ChatAgent(
            system_message=system_prompt,
            model=web_agent_model,
            output_language=self.output_language,
        )

        planning_system_prompt = """
您是一个优秀的规划代理,可以协助用户规划需要多步浏览器交互的复杂任务。
        """

        planning_agent = ChatAgent(
            system_message=planning_system_prompt,
            model=planning_model,
            output_language=self.output_language,
        )

        return web_agent, planning_agent

    def _observe(
        self, task_prompt: str, detailed_plan: Optional[str] = None
    ) -> Tuple[str, str, str]:
        r"""Let agent observe the current environment, and get the next action."""

        detailed_plan_prompt = ""

        if detailed_plan is not None:
            detailed_plan_prompt = f"""
这里是一个关于如何逐步解决任务的计划，你必须遵循:
<detailed_plan>{detailed_plan}<detailed_plan>
        """

        # 获取当前状态截图和可交互元素
        som_screenshot, _, interactive_elements = self.browser.get_som_screenshot(save_image=True)
        img = _reload_image(som_screenshot)

        # 提取元素ID和类型信息
        elements_info = []
        for element_id, element_data in interactive_elements.items():
            elements_info.append(int(element_id.strip()))

        # 将元素信息添加到提示中
        elements_prompt = f"""
当前页面上的可交互元素id列表：
{elements_info}
"""

        # 更新提示词
        observe_prompt = f"""
        您是一个网页代理，需帮助我完成以下高级任务：
        <task>{task_prompt}</task>
        当前，我已基于浏览器状态制作了截图（仅当前视口，非整个网页），并标记了交互元素。
        请根据任务要求和当前视口状态，提供下一个适当操作。

        {detailed_plan_prompt}

        ### 当前可用信息
        - **交互元素**：{elements_prompt}
        - **浏览器功能**：{AVAILABLE_ACTIONS_PROMPT}
        - **最近操作历史**（最多 {self.history_window} 条）：
          <history>{self.history[-self.history_window:]}</history>

        ### 输出要求
        以 JSON 格式返回以下字段：
        - `observation`：当前视口的详细描述，基于截图观察，避免对历史操作过度假设。
        - `reasoning`：下一步操作的推理，包括可能障碍及解决思路，参考历史避免重复错误。
        - `action_code`：具体操作代码，仅一步，不含注释。

        #### 输出示例
```json
        {{
            "observation": "当前视口显示亚马逊搜索页面，包含输入框和搜索按钮。",
            "reasoning": "任务要求搜索产品，需先填写搜索词并提交。",
            "action_code": "fill_input_id(1, 'laptop')"
        }}
        {{
            "observation": [图像描述],
            "reasoning": [您的推理],
            "action_code": "fill_input_id([ID], [TEXT])"
        }}
```

操作指南：
1.任务导向：始终围绕 {task_prompt} 推进，基于当前视口选择操作。
2.ID 准确性：仅使用 {elements_prompt} 中提供的交互元素 ID。
3.页面检查：
若操作（如 click_id）后页面未变，参考历史中的 success 判断是否成功。
未找到答案时，使用 scroll_up() 或 scroll_down() 检查完整页面。
4.错误处理：
若方法无效，使用 back() 回退，尝试其他路径（如点击链接）。
遇到人工验证时，使用 back() 规避。
5.复杂场景：
处理维基百科修订历史时，先调整单页显示条目至最大，再用 find_text_on_page 定位关键信息。
灵活使用下拉菜单筛选内容。
6.停止条件：
仅在所有方法无效时使用 stop()，并说明问题。
注意事项：
1.操作必须基于当前视口 ID，而非历史记录。
2.避免重复执行相同操作，检查历史以优化决策。
3.若无法解决，报告具体困难并停止。  
        """

        # 将截图和更新后的提示发送给模型
        message = BaseMessage.make_user_message(
            role_name='user', content=observe_prompt, image_list=[img]
        )
        # 重置web_agent的历史消息 重要的历史信息已经包含在observe_prompt中的history字段
        self.web_agent.reset()
        resp = self.web_agent.step(message)

        resp_content = resp.msgs[0].content

        resp_dict = _parse_json_output(resp_content)
        observation_result: str = resp_dict.get("observation", "")
        reasoning_result: str = resp_dict.get("reasoning", "")
        action_code: str = resp_dict.get("action_code", "")

        if action_code and "(" in action_code and ")" not in action_code:
            action_match = re.search(
                r'"action_code"\s*:\s*[`"]([^`"]*\([^)]*\))[`"]', resp_content
            )
            if action_match:
                action_code = action_match.group(1)
            else:
                logger.warning(
                    f"Incomplete action_code detected: {action_code}"
                )
                if action_code.startswith("fill_input_id("):
                    parts = action_code.split(",", 1)
                    if len(parts) > 1:
                        id_part = (
                            parts[0].replace("fill_input_id(", "").strip()
                        )
                        action_code = f"fill_input_id({id_part}, 'Please fill the text here.')"

        action_code = action_code.replace("`", "").strip()

        return observation_result, reasoning_result, action_code

    def _act(self, action_code: str) -> Tuple[bool, str]:
        r"""Let agent act based on the given action code.
        Args:
            action_code (str): The action code to act.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating whether
                the action was successful, and the information to be returned.
        """

        def _check_if_with_feedback(action_code: str) -> bool:
            r"""Check if the action code needs feedback."""

            for action_with_feedback in ACTION_WITH_FEEDBACK_LIST:
                if action_with_feedback in action_code:
                    return True

            return False

        def _fix_action_code(action_code: str) -> str:
            r"""Fix potential missing quotes in action code"""

            match = re.match(r'(\w+)\((.*)\)', action_code)
            if not match:
                return action_code

            func_name, args_str = match.groups()

            args = []
            current_arg = ""
            in_quotes = False
            quote_char = None

            for char in args_str:
                if char in ['"', "'"]:
                    if not in_quotes:
                        in_quotes = True
                        quote_char = char
                        current_arg += char
                    elif char == quote_char:
                        in_quotes = False
                        quote_char = None
                        current_arg += char
                    else:
                        current_arg += char
                elif char == ',' and not in_quotes:
                    args.append(current_arg.strip())
                    current_arg = ""
                else:
                    current_arg += char

            if current_arg:
                args.append(current_arg.strip())

            fixed_args = []
            for arg in args:
                if (
                    (arg.startswith('"') and arg.endswith('"'))
                    or (arg.startswith("'") and arg.endswith("'"))
                    or re.match(r'^-?\d+(\.\d+)?$', arg)
                    or re.match(r'^-?\d+\.?\d*[eE][-+]?\d+$', arg)
                    or re.match(r'^0[xX][0-9a-fA-F]+$', arg)
                ):
                    fixed_args.append(arg)

                else:
                    fixed_args.append(f"'{arg}'")

            return f"{func_name}({', '.join(fixed_args)})"

        action_code = _fix_action_code(action_code)
        prefix = "self.browser."
        code = f"{prefix}{action_code}"

        try:
            if _check_if_with_feedback(action_code):
                # execute code, and get the executed result
                result = eval(code)
                time.sleep(1)
                return True, result

            else:
                exec(code)
                time.sleep(1)
                return True, "Action was successful."

        except Exception as e:
            time.sleep(1)
            return (
                False,
                f"Error while executing the action {action_code}: {e}. "
                f"If timeout, please recheck whether you have provided the "
                f"correct identifier.",
            )

    def _get_final_answer(self, task_prompt: str) -> str:
        prompt = f"""
我们正在解决一个需要多步浏览器交互的复杂网页任务。经过多步观察、推理和与浏览器的交互后，我们认为任务已经完成。
以下是我们执行过的所有轨迹：
<history>{self.history}</history>
请找出最终答案，或者提供有价值的见解和发现（例如，如果之前的操作包含下载文件，您的输出应该包含下载文件的路径）关于整体任务：<task>{task_prompt}</task>
        """
        message = BaseMessage.make_user_message(
            role_name='user',
            content=prompt,
        )

        resp = self.web_agent.step(message)
        return resp.msgs[0].content

    def _make_reflection(self, task_prompt: str) -> str:
        reflection_prompt = f"""
现在我们正在处理一个需要多步浏览器交互的复杂任务。任务是：<task>{task_prompt}</task>
为了实现这个目标，我们已经进行了一系列的观察、推理和操作。我们也对之前的状态进行了反思。

以下是我们可以使用的全局浏览器功能：
{AVAILABLE_ACTIONS_PROMPT}

以下是我们最近执行的{self.history_window}个轨迹（最多）：
<history>{self.history[-self.history_window:]}</history>

提供的图像是浏览器的当前状态，我们已经标记了交互元素。
请仔细检查任务要求和浏览器的当前状态，然后对之前的步骤进行反思，思考它们是否有帮助以及原因，提供详细的反馈和对下一步的建议。

您的输出应该是json格式，包含以下字段：
- `reflection`: 关于之前步骤的反思，思考它们是否有帮助以及原因，提供详细的反馈。
- `suggestion`: 对下一步的建议，提供详细的建议，包括基于浏览器当前状态的整体任务的常见解决方案。
        """
        som_image, _ = self.browser.get_som_screenshot()
        img = _reload_image(som_image)

        message = BaseMessage.make_user_message(
            role_name='user', content=reflection_prompt, image_list=[img]
        )

        resp = self.web_agent.step(message)

        return resp.msgs[0].content

    def _task_planning(self, task_prompt: str, start_url: str) -> str:
        planning_prompt = f"""
<task>{task_prompt}</task>
根据上述问题，假设我们通过浏览器进行交互，在访问初始网页 `{start_url}` 后，一般的交互过程是怎样的？  
请注意，需考虑部分可观察马尔可夫决策过程（POMDP）的特性，即网页的状态只能被部分观察到，因此信息是不完全的。  
在回答时，请避免对你的计划表现出过度自信。  
请按照以下步骤作答：  
1. 首先详细重述任务，确保对问题的理解准确无误；  
2. 然后提供一个详细的解决方案计划，描述交互的具体步骤和方法。
        """

        message = BaseMessage.make_user_message(
            role_name='user', content=planning_prompt
        )

        resp = self.planning_agent.step(message)
        return resp.msgs[0].content

    def _task_replanning(
        self, task_prompt: str, detailed_plan: str
    ) -> Tuple[bool, str]:
        replanning_prompt = f"""
我们正在使用浏览器交互来解决一个需要多步操作的复杂任务。
以下是整体任务：
<overall_task>{task_prompt}</overall_task>

为了解决这个任务，我们之前制定了一个详细计划。以下是该计划：
<detailed_plan>{detailed_plan}</detailed_plan>

根据上述任务，我们已经进行了一系列观察、推理和操作。以下是我们最近执行的{self.history_window}个轨迹（最多）：
<history>{self.history[-self.history_window:]}</history>

然而，任务尚未完成。由于任务是部分可观察的，如果必要的话，我们可能需要根据浏览器的当前状态重新规划任务。
现在请仔细检查当前的任务规划方案和我们的历史操作，然后判断任务是否需要从根本上重新规划。如果需要，请提供详细的重新规划方案（包括重述的整体任务）。

您的输出应该是json格式，包含以下字段：
- `if_need_replan`: bool, 一个布尔值，表示任务是否需要从根本上重新规划。
- `replanned_schema`: str, 任务的重新规划方案，与原方案相比不应该有太大变化。如果任务不需要重新规划，该值应为空字符串。
        """

        # 重置 planning_agent 的历史消息 重要的历史信息已经包含在replanning_prompt中的history字段
        self.planning_agent.reset()
        resp = self.planning_agent.step(replanning_prompt)
        resp_dict = _parse_json_output(resp.msgs[0].content)

        if_need_replan = resp_dict.get("if_need_replan", False)
        replanned_schema = resp_dict.get("replanned_schema", "")

        if if_need_replan:
            return True, replanned_schema
        else:
            return False, replanned_schema

    @dependencies_required("playwright")
    def browse_url(
        self, task_prompt: str, start_url: str, round_limit: int = 12
    ) -> str:
        r"""A powerful toolkit which can simulate the browser interaction to solve the task which needs multi-step actions.

        Args:
            task_prompt (str): The task prompt to solve.
            start_url (str): The start URL to visit.
            round_limit (int): The round limit to solve the task.
                (default: :obj:`12`).

        Returns:
            str: The simulation result to the task.
        """

        self._reset()
        task_completed = False
        detailed_plan = self._task_planning(task_prompt, start_url)
        logger.debug(f"Detailed plan: {detailed_plan}")

        self.browser.init()
        self.browser.visit_page(start_url)

        for i in range(round_limit):
            observation, reasoning, action_code = self._observe(
                task_prompt, detailed_plan
            )
            logger.debug(f"Observation: {observation}")
            logger.debug(f"Reasoning: {reasoning}")
            logger.debug(f"Action code: {action_code}")

            if "stop" in action_code:
                task_completed = True
                trajectory_info = {
                    "round": i,
                    "observation": observation,
                    "thought": reasoning,
                    "action": action_code,
                    "action_if_success": True,
                    "info": None,
                    "current_url": self.browser.get_url(),
                }
                self.history.append(trajectory_info)
                break

            else:
                success, info = self._act(action_code)
                if not success:
                    logger.warning(f"Error while executing the action: {info}")

                trajectory_info = {
                    "round": i,
                    "observation": observation,
                    "thought": reasoning,
                    "action": action_code,
                    "action_if_success": success,
                    "info": info,
                    "current_url": self.browser.get_url(),
                }
                self.history.append(trajectory_info)

                # replan the task if necessary
                if_need_replan, replanned_schema = self._task_replanning(
                    task_prompt, detailed_plan
                )
                if if_need_replan:
                    detailed_plan = replanned_schema
                    logger.debug(f"Replanned schema: {replanned_schema}")

        if not task_completed:
            simulation_result = f"""
                任务未在轮次限制内完成。请检查最后{self.history_window}轮的信息，看看是否有任何有用的信息：
                <history>{self.history[-self.history_window:]}</history>
            """

        else:
            simulation_result = self._get_final_answer(task_prompt)

        self.browser.close()
        return simulation_result

    def get_tools(self) -> List[FunctionTool]:
        return [FunctionTool(self.browse_url)]
