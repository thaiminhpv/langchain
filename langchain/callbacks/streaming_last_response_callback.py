"""Callback Handler streams callback on new llm token in last agent response."""
from typing import Any, List, Optional, Callable, Union

import warnings
from queue import Queue

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentFinish
from langchain.schema import OutputParserException
from langchain.agents.agent_types import AgentType

def create_streaming_callback(
    agent: Optional[AgentType] = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    callback_func: Optional[Callable[[str], None]] = None,
    **kwargs,
) -> "StreamingLastResponseCallbackHandler":
    """Create a callback handler for streaming in agents."""
    if agent == AgentType.ZERO_SHOT_REACT_DESCRIPTION:
        return StreamingLastResponseCallbackHandler(
            answer_prefix_phrases=[
                "Final Answer:",
            ],
            callback_func=callback_func,
            **kwargs
        )
    elif agent == AgentType.CONVERSATIONAL_REACT_DESCRIPTION:
        return StreamingLastResponseCallbackHandler(
            answer_prefix_phrases=[
                "Do I need to use a tool? No\nAI:",
                "Do I need to use a tool? No",
            ],
            error_stop_streaming_phrases=[
                "Do I need to use a tool? No\nAction:",
            ],
            callback_func=callback_func,
            **kwargs
        )

    elif agent == AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION:
        # TODO: Post processing remove last '"\n}' after final answer
        raise NotImplementedError
        return StreamingLastResponseCallbackHandler(
            answer_prefix_phrases=[
                'Final Answer",\n    "action_input": "',
                'Final Answer",\n  "action_input": "',
            ],
            callback_func=callback_func,
            **kwargs
        )
    elif agent == AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION:
        # TODO: Post processing remove last '"\n}\n```' after final answer
        raise NotImplementedError
        return StreamingLastResponseCallbackHandler(
            answer_prefix_phrases=[
                'Final Answer",\n    "action_input": "',
                'Final Answer",\n  "action_input": "',
            ],
            callback_func=callback_func,
            **kwargs
        )
    else:
        raise NotImplementedError

        

class StreamingLastResponseCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming in agents.
    Only works with agents using LLMs that support streaming.

    Only the final output of the agent will be streamed.

    Example:
        .. code-block:: python

            # Callback function to print the next token
            from langchain.agents import load_tools, initialize_agent, AgentType
            from langchain.llms import OpenAI
            from langchain.callbacks.streaming_last_response_callback import create_streaming_callback

            def _callback(next_token: str):
                if next_token is StopIteration:
                    print("\n[Done]")
                    return
                else:
                    print(next_token, end="", flush=True)
            streaming_callback = create_streaming_callback(agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, callback_func=_callback)
            llm = OpenAI(temperature=0, streaming=True)
            tools = load_tools(["serpapi", "llm-math"], llm=llm)
            agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False)
            answer = agent.run("Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?", callbacks=[streaming_callback])

            # Or use as iterator
            from langchain.agents import load_tools, initialize_agent, AgentType
            from langchain.llms import OpenAI
            from langchain.callbacks.streaming_last_response_callback import create_streaming_callback
            import threading

            stream = create_streaming_callback(agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
            llm = OpenAI(temperature=0, streaming=True)
            tools = load_tools(["serpapi", "llm-math"], llm=llm)
            agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False)

            def _run():
                agent.run("Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?", callbacks=[stream])
            
            threading.Thread(target=_run).start()
            
            for token in stream:
                print(token, end="", flush=True)
    """
    
    def __init__(self,
        answer_prefix_phrases: Optional[List[str]] = ["Final Answer:"],
        error_stop_streaming_phrases: Optional[List[str]] = [],
        callback_func: Optional[Callable[[str], None]] = None,
        postprocess_func: Optional[Callable[[List[str]], List[str]]] = None,
        postprocess_window_size: Optional[int] = 3,
        postprocess_sliding_window_step: Optional[int] = 1,
        output_stream_prefix: bool = False,
        tiktoken_encoding: Optional[str] = "cl100k_base",
    )-> None:
        """
        Args:
            answer_prefix_phrases: List of phrases that indicate that the next
                token is the final answer. Multiple phrases are allowed, the first
                one that matches will be used. Phrase matching is case sensitive.
                Some phrases can be a substring of the other phrase. If multiple
                phrases are detected, the longest one will be used.
                Example: ["Final Answer:", "Final Answer"]
            error_stop_streaming_phrases: List of phrases that indicate that the
                next token is an error message and that the streaming should stop.
                Multiple phrases are allowed, the first one that matches will stop
                the streaming. Phrase matching is case sensitive.
            callback_func: Callback function to call when a new token is available
                after the answer_prefix_phrases. The callback function should take
                a single argument, which is the next token. If the last token is
                reached, the callback function will be called with StopIteration.
            postprocess_func: On-the-fly postprocess function to apply to the output
                stream. The postprocess function should take a single argument, which
                is the output stream as a list of tokens. The postprocess function
                should return a list of tokens to be replaced in the output stream.
                If None, no postprocessing will be applied.
            postprocess_window_size: The window size to use for the postprocess_func.
                The actual used window size will be the maximum of postprocess_window_size,
                max length of answer_prefix_phrases, and error_stop_streaming_phrases.
            postprocess_sliding_window_step: Default is 1. This means that the
                postprocess_func will be applied to the detection queue after
                every new token.
            output_stream_prefix: If True, the output stream will include the
                found answer_prefix_phrases. If False, the output stream will
                only include the final answer and exclude the matched
                answer_prefix_phrases.
            tiktoken_encoding: The encoding to use for the tiktoken. Default is
                "cl100k_base". See langchain.schema.TikToken for more details.
        """
        super().__init__()

        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "Could not import tiktoken python package. "
                "This is needed in order to calculate detection_windows_size for StreamingLastResponseCallbackHandler"
                "Please install it with `pip install tiktoken`."
            )
        self._enc = tiktoken.get_encoding(tiktoken_encoding)

        # sort by length, so that the longest phrase will be detected first.
        self.answer_prefix_phrases = sorted(answer_prefix_phrases, key=len, reverse=True)
        self.error_stop_streaming_phrases = sorted(error_stop_streaming_phrases, key=len, reverse=True)

        if answer_prefix_phrases:
            _max_answer_prefix_phrases_token_len = max(len(self._enc.encode(_answer_prefix_phrase)) for _answer_prefix_phrase in self.answer_prefix_phrases)
        else:
            _max_answer_prefix_phrases_token_len = 1
        if error_stop_streaming_phrases:
            _max_error_stop_streaming_phrases_token_len = max(len(self._enc.encode(_error_stop_streaming_phrase)) for _error_stop_streaming_phrase in self.error_stop_streaming_phrases)
        else:
            _max_error_stop_streaming_phrases_token_len = 1

        # do not use Queue(maxsize=...), because it will block the queue.
        self.detection_queue_size = max(_max_answer_prefix_phrases_token_len, _max_error_stop_streaming_phrases_token_len, postprocess_window_size)

        self.detection_queue: Queue[str] = Queue()
        self.output_queue: Queue[str] = Queue()

        self.is_streaming_answer = False # If the answer is reached, the streaming will be started.
        self.postprocess_func = postprocess_func
        self.postprocess_sliding_window_step = postprocess_sliding_window_step
        self.step_counter = 0
        self.output_stream_prefix = output_stream_prefix

        if callback_func is not None:
            self.callback_func = callback_func
        else:
            self.callback_func = lambda new_token: None
               
    def __iter__(self):
        """
        This function is used when the callback handler is used as an iterator.
        """
        while True:
            token = self._pop_out_queue()
            if token is StopIteration: break
            yield token
        
    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """
        This function is called when the agent finishes. It will flush the detection queue when there are no more tokens from on_llm_new_token.
        """
        super().on_agent_finish(finish, **kwargs)
        self._flush_detection_queue()

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """
        This function is called when a new token is generated by the LLM.
        """
        self.step_counter += 1
        self.detection_queue.put(token)

        if self.is_streaming_answer:
            # if the answer is reached, the streaming will be started.
            last_token = None
            if self.detection_queue.qsize() > self.detection_queue_size:
                if self.step_counter % self.postprocess_sliding_window_step == 0:
                    self._post_process_detection_queue()
                    self._check_abnormal_in_detection_queue()
                last_token = self.detection_queue.get()
                self._callback(last_token)
                
        elif self.detection_queue.qsize() > self.detection_queue_size:
            # if the answer is not reached, the detection queue will be checked.
            _answer_prefix_phrase = self._check_if_answer_reached()
            if _answer_prefix_phrase is not None:
                # remove all answer prefix tokens from the detection queue
                for _ in range(len(self._enc.encode(_answer_prefix_phrase))):
                    _token = self.detection_queue.get()
                    if self.output_stream_prefix:
                        # output the answer prefix token
                        self._callback(_token)
            else:
                # if the answer is not reached, the detection queue will pop out the oldest token.
                self.detection_queue.get()
     
    def _post_process_detection_queue(self) -> None:
        """
        Post process on-the-fly the detection queue by using used-defined postprocess_func.
        This function will be called every postprocess_sliding_window_step.
        """
        if self.postprocess_func is not None:
            tokens = list(self.detection_queue.queue)
            tokens = self.postprocess_func(tokens)
            self.detection_queue.queue.clear()
            for token in tokens:
                self.detection_queue.put(token)
    
    def _check_abnormal_in_detection_queue(self) -> None:
        """
        Check if the detection queue is abnormal. If the detection queue is abnormal, it will raise OutputParserException and stop the streaming.
        Check by using error_stop_streaming_phrases. If the error_stop_streaming_phrases is detected, the streaming will be stopped.
        """
        sentence = "".join(self.detection_queue.queue)

        for error_stop_streaming_phrases in self.error_stop_streaming_phrases:
            if error_stop_streaming_phrases in sentence:
                self._callback(OutputParserException(f"Abnormal in detection queue detected. Detection queue: '{self.detection_queue.queue}'. Abnormal: '{error_stop_streaming_phrases}'"))
    
    def _flush_detection_queue(self):
        """
        Flush detection queue. This will be called when the agent is finished to flush all the remaining tokens in the detection queue.
        """
        while not self.detection_queue.empty():
            if not self.is_streaming_answer:
                _answer_prefix_phrase = self._check_if_answer_reached()
                if _answer_prefix_phrase is not None:
                    # remove all answer prefix tokens from detection queue
                    if not self.output_stream_prefix:
                        for _ in range(len(self._enc.encode(_answer_prefix_phrase))):
                            while self.detection_queue.queue[0] == '':
                                self.detection_queue.get()
                            self.detection_queue.get()
                    else:
                        for _ in range(len(self._enc.encode(_answer_prefix_phrase))):
                            self._callback(self.detection_queue.get())
                else:
                    self.detection_queue.get()
            else:
                self._callback(self.detection_queue.get())
        
        if not self.is_streaming_answer:
            warnings.warn("StreamingLastResponseCallbackHandler is not streaming answer, but agent_finish is called.")
            self.output_queue.put(StopIteration)

        self._callback(StopIteration)
    
    def _callback(self, text: Union[str, Exception]) -> None:
        """
        Callback function. It will put the text to the output queue, and call the callback_func.
        """
        if text is StopIteration:
            self.output_queue.put(text)
            self.callback_func(text)
        elif isinstance(text, Exception):
            self.output_queue.put(text)
            raise text
        else:
            self.output_queue.put(text)
            self.callback_func(text)
     
    def _check_if_answer_reached(self) -> Optional[str]:
        """
        Check if the answer is reached. If the answer is reached, it will return the answer prefix phrase.
        If the answer is not reached, it will return None.
        """
        if self.detection_queue.queue[0] == '':
            return None
        for _answer_prefix_str in self.answer_prefix_phrases:
            current_output = "".join(self.detection_queue.queue)
            if current_output.strip().startswith(_answer_prefix_str):
                self.is_streaming_answer = True
                return _answer_prefix_str
        return None

    def _pop_out_queue(self) -> str:
        """
        Pop out the output queue. If the output queue is empty, it will wait until the output queue is not empty.
        """
        token = self.output_queue.get()
        if isinstance(token, Exception):
            raise token
        return token
 