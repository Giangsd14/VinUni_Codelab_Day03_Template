"""
Lab #3: Baseline Chatbot vs ReAct Agent
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.
"""

import json
from tools import TOOL_DEFINITIONS, TOOL_MAP, get_flight_info, get_weather_forecast

SYSTEM_PROMPT = """Bạn là một ReAct Agent thông minh hỗ trợ khách hàng Vingroup.
Bạn chỉ sử dụng các công cụ sau:
{tools}

Quy trình trả lời bắt buộc:
Thought: <Suy nghĩ bước tiếp theo>
Action: {{"name": "<tên tool>", "args": {{<tham số>}}}}
Observation: <Kết quả từ tool>
... (Lặp lại cho tới khi có đủ dữ liệu)
Final Answer: <Câu trả lời hoàn chỉnh cho khách hàng>

Quy tắc xử lý lỗi (bắt buộc):
- Nếu Observation trả về "error" hoặc "Invalid JSON format", hãy thử lại với tham số đúng.
- Nếu gặp lỗi từ cùng một tool >= 2 lần liên tiếp, DỪNG và đưa ra Final Answer thông báo lỗi lịch sự cho khách hàng.
"""

class ChatbotBaseline:
    """Baseline LLM Chatbot (Không sử dụng ReAct Loop hay Tools)"""

    def query(self, user_input: str) -> str:
        """
        Baseline: gọi thẳng tool functions một lần (không có Thought-Action loop),
        rồi tổng hợp câu trả lời tiếng Việt.
        """
        # --- Bước 1: Lấy thông tin chuyến bay HAN -> SGN dưới 2 triệu ---
        flights = get_flight_info(origin="HAN", destination="SGN", max_price=2_000_000)

        if flights:
            flight_lines = []
            for fl in flights:
                flight_lines.append(
                    f"  • {fl['flight_number']} ({fl['airline']}) – "
                    f"Khởi hành {fl['departure_time']} – "
                    f"Giá: {fl['price_vnd']:,} VND"
                )
            flight_info = "\n".join(flight_lines)
        else:
            flight_info = "  Không tìm thấy chuyến bay phù hợp."

        # --- Bước 2: Lấy thời tiết SGN để gợi ý trang phục ---
        weather = get_weather_forecast(city_code="SGN")

        if "error" not in weather:
            weather_info = (
                f"  Thành phố: {weather.get('city', 'SGN')}\n"
                f"  Nhiệt độ : {weather.get('temperature_c', '?')}°C – {weather.get('condition', '')}\n"
                f"  Độ ẩm   : {weather.get('humidity_pct', '?')}%\n"
                f"  Gợi ý   : {weather.get('recommendation', '')}"
            )
        else:
            weather_info = f"  Lỗi dữ liệu thời tiết: {weather['error']}"

        # --- Tổng hợp câu trả lời ---
        answer = (
            f"[Chatbot Baseline] Câu hỏi: {user_input}\n\n"
            f"✈️  Chuyến bay HAN → SGN dưới 2.000.000 VND:\n{flight_info}\n\n"
            f"🌤️  Thời tiết tại SGN & gợi ý trang phục:\n{weather_info}"
        )
        return answer

class ReActAgent:
    """ReAct Agent có sử dụng Thought-Action-Observation Loop"""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace = []

    # ------------------------------------------------------------------
    # Hàm nội bộ: giả lập phần "suy luận" của Agent (thay cho LLM call)
    # Trả về chuỗi văn bản theo chuẩn Thought / Action / Final Answer
    # ------------------------------------------------------------------
    def _think(self, user_input: str, history: list) -> str:
        """
        Sinh ra bước suy luận tiếp theo dựa trên lịch sử đã có.
        Trong bài lab này, logic suy luận được viết tường minh (rule-based)
        để minh họa cơ chế ReAct mà không cần gọi LLM thực.
        """
        # Kiểm tra xem tool nào đã được gọi
        called_tools = [step.get("tool") for step in history if step.get("type") == "observation"]

        # ----- Bước 1: Chưa tìm chuyến bay → gọi get_flight_info -----
        if "get_flight_info" not in called_tools:
            return (
                "Thought: Người dùng hỏi về chuyến bay HAN→SGN dưới 2 triệu. "
                "Tôi cần gọi get_flight_info để lấy danh sách chuyến bay.\n"
                'Action: {"name": "get_flight_info", "args": {"origin": "HAN", "destination": "SGN", "max_price": 2000000}}'
            )

        # ----- Bước 2: Đã có chuyến bay, chưa lấy thời tiết → gọi get_weather_forecast -----
        if "get_weather_forecast" not in called_tools:
            return (
                "Thought: Đã có thông tin chuyến bay. "
                "Tiếp theo cần lấy thời tiết SGN để gợi ý trang phục.\n"
                'Action: {"name": "get_weather_forecast", "args": {"city_code": "SGN"}}'
            )

        # ----- Bước 3: Đã có đủ dữ liệu → tổng hợp Final Answer -----
        # Thu thập kết quả từ trace
        flight_data = next(
            (s["result"] for s in history if s.get("type") == "observation" and s.get("tool") == "get_flight_info"),
            []
        )
        weather_data = next(
            (s["result"] for s in history if s.get("type") == "observation" and s.get("tool") == "get_weather_forecast"),
            {}
        )

        # Định dạng chuyến bay
        if flight_data:
            flight_lines = [
                f"  • {fl['flight_number']} ({fl['airline']}) – "
                f"Khởi hành {fl['departure_time']} – Giá: {fl['price_vnd']:,} VND"
                for fl in flight_data
            ]
            flight_str = "\n".join(flight_lines)
        else:
            flight_str = "  Không tìm thấy chuyến bay phù hợp."

        # Định dạng thời tiết
        if weather_data and "error" not in weather_data:
            weather_str = (
                f"  {weather_data.get('city', 'SGN')} – "
                f"{weather_data.get('temperature_c', '?')}°C, {weather_data.get('condition', '')}, "
                f"độ ẩm {weather_data.get('humidity_pct', '?')}%.\n"
                f"  Gợi ý trang phục: {weather_data.get('recommendation', '')}"
            )
        else:
            weather_str = "  Không có dữ liệu thời tiết."

        return (
            "Thought: Đã có đủ thông tin về chuyến bay và thời tiết. Tổng hợp câu trả lời.\n"
            f"Final Answer:\n"
            f"✈️  Chuyến bay HAN → SGN dưới 2.000.000 VND:\n{flight_str}\n\n"
            f"🌤️  Thời tiết SGN & gợi ý trang phục:\n{weather_str}"
        )

    # ------------------------------------------------------------------
    # Vòng lặp chính: Thought → Action → Observation → … → Final Answer
    # ------------------------------------------------------------------
    def run(self, user_input: str) -> str:
        # TODO 1: Khởi tạo mảng lưu lịch sử conversation / traces
        self.trace = []
        history = []   # lưu các bước observation trong phiên hiện tại

        self.trace.append({"step": "init", "type": "user_input", "user_input": user_input})
        print(f"\n[ReAct] Câu hỏi: {user_input}\n")

        iteration = 0
        # Trap 3: đếm số lần lỗi liên tiếp để tránh vòng lặp vô tận
        consecutive_errors = 0
        MAX_CONSECUTIVE_ERRORS = 2

        # TODO 2: Thiết lập vòng lặp while iteration < self.max_iterations
        while iteration < self.max_iterations:
            iteration += 1
            print(f"--- Iteration {iteration} ---")

            # TODO 3: Phân tích Thought / Action từ Agent
            agent_output = self._think(user_input, history)
            print(agent_output)

            # Tách Thought ra khỏi output
            thought = ""
            if agent_output.startswith("Thought:"):
                thought = agent_output.split("\n")[0].replace("Thought:", "").strip()

            # Kiểm tra Final Answer
            if "Final Answer:" in agent_output:
                final_answer = agent_output.split("Final Answer:", 1)[1].strip()
                self.trace.append({
                    "step": f"iteration_{iteration}",
                    "type": "final_answer",
                    "thought": thought,
                    "answer": final_answer,
                })
                return final_answer

            # TODO 4: Thực thi Tool trong TOOL_MAP nếu có Action
            action_str = None
            for line in agent_output.split("\n"):
                if line.startswith("Action:"):
                    action_str = line.replace("Action:", "").strip()
                    break

            if action_str:
                # ── Trap 2: parse JSON trong try/except, phản hồi lỗi format ──
                try:
                    action = json.loads(action_str)
                except json.JSONDecodeError as e:
                    invalid_obs = f"Observation: Invalid JSON format – {e}. Hãy trả về Action đúng dạng JSON."
                    print(invalid_obs)
                    consecutive_errors += 1
                    self.trace.append({
                        "step": f"iteration_{iteration}",
                        "type": "error",
                        "kind": "invalid_json",
                        "raw_action": action_str,
                        "message": invalid_obs,
                    })
                    # Trap 3: nếu lỗi liên tiếp quá ngưỡng → buộc Final Answer
                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        error_answer = (
                            "Xin lỗi, hệ thống gặp sự cố xử lý yêu cầu của bạn "
                            f"(lỗi định dạng Action liên tiếp {consecutive_errors} lần). "
                            "Vui lòng thử lại sau."
                        )
                        self.trace.append({
                            "step": f"iteration_{iteration}",
                            "type": "final_answer",
                            "thought": "Lỗi liên tiếp vượt ngưỡng, buộc kết thúc.",
                            "answer": error_answer,
                        })
                        return error_answer
                    # Cho agent thêm một lượt để tự sửa
                    history.append({
                        "step": f"iteration_{iteration}",
                        "type": "observation",
                        "tool": None,
                        "result": {"error": f"Invalid JSON format: {e}"},
                    })
                    continue

                tool_name_raw = action.get("name", "")
                tool_args = action.get("args", {})

                # ── Trap 1: chuẩn hoá tên tool (bỏ khoảng trắng, viết thường) ──
                tool_name = tool_name_raw.strip().lower()

                # Ghi bước Action vào trace
                self.trace.append({
                    "step": f"iteration_{iteration}",
                    "type": "action",
                    "thought": thought,
                    "tool": tool_name,
                    "args": tool_args,
                })

                # Gọi tool từ TOOL_MAP
                if tool_name in TOOL_MAP:
                    tool_result = TOOL_MAP[tool_name](**tool_args)
                else:
                    tool_result = {"error": f"Tool '{tool_name}' không tồn tại trong TOOL_MAP."}

                observation_str = json.dumps(tool_result, ensure_ascii=False, indent=2)
                print(f"Observation: {observation_str}\n")

                # ── Trap 3: kiểm tra error trong kết quả tool ──
                if isinstance(tool_result, dict) and "error" in tool_result:
                    consecutive_errors += 1
                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        error_answer = (
                            f"Xin lỗi, công cụ '{tool_name}' gặp lỗi liên tiếp "
                            f"{consecutive_errors} lần: {tool_result['error']}. "
                            "Vui lòng thử lại hoặc liên hệ hỗ trợ."
                        )
                        self.trace.append({
                            "step": f"iteration_{iteration}",
                            "type": "observation",
                            "tool": tool_name,
                            "result": tool_result,
                        })
                        self.trace.append({
                            "step": f"iteration_{iteration}",
                            "type": "final_answer",
                            "thought": "Lỗi tool liên tiếp vượt ngưỡng, buộc kết thúc.",
                            "answer": error_answer,
                        })
                        return error_answer
                else:
                    # Reset đếm lỗi khi tool chạy thành công
                    consecutive_errors = 0

                # TODO 5: Ghi lại Observation và lặp lại cho tới khi ra Final Answer
                step_record = {
                    "step": f"iteration_{iteration}",
                    "type": "observation",
                    "tool": tool_name,
                    "result": tool_result,
                }
                self.trace.append(step_record)
                history.append(step_record)

            else:
                # Không có Action và không có Final Answer → dừng vòng lặp
                self.trace.append({
                    "step": f"iteration_{iteration}",
                    "type": "error",
                    "message": "Không tìm thấy Action hoặc Final Answer trong output.",
                })
                break

        return "[ReAct Agent] Đã đạt giới hạn vòng lặp mà không có Final Answer."

def main():
    user_query = "Tìm chuyến bay từ HAN đi SGN dưới 2 triệu, và thời tiết SGN nên mặc gì?"
    # user_query = "Có chuyến bay nào từ HAN đi DAD giá dưới 1.5 triệu không?"
    
    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))
    
    print("\n=== RUNNING REACT AGENT ===")
    agent = ReActAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result)
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()

