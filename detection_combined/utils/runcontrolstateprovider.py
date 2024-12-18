#!/usr/bin/env python3

from typing import List
import os
import time as t
import datetime as dt
import asyncio as aio
import logging

import pandas as pd
from rich import print

from .beautysingleton import BeautySingleton
from .shiftedtimesingleton import ShiftedTimeSingleton

_run_state_vars = ['ATLAS', 'RCStateInfo', 'state', 'RunCtrl.RootController']


class RunControlStateProvider():

    def __init__(self,
                    polling_interval: dt.timedelta = dt.timedelta(seconds=10),
                    pbeast_server: str = 'https://atlasop.cern.ch') -> None:

        self._polling_interval = polling_interval
        self._pbeast_server = pbeast_server

        os.environ['PBEAST_SERVER_SSO_SETUP_TYPE'] = 'AutoUpdateKerberos'

        self._beauty_instance = BeautySingleton(server=self._pbeast_server).instance()

        self._timestamp_last = None

        self._logger = logging.getLogger(__name__)
    

    async def wait_for_states(self,
                                target_states: List[str],
                                return_delay: dt.timedelta) -> pd.DataFrame:

        while True:

            time_start = t.monotonic()

            request_time = dt.datetime.now()
            
            request_time = ShiftedTimeSingleton(dt.datetime(2023, 7, 13, 14, 30, 0)).now()

            # print(request_time)

            self._logger.debug(f'Requesting Run Control status '
                                f'from PBEAST at timestamp {request_time}')
            self._logger.debug(f'Request vars: {_run_state_vars[0]}, {_run_state_vars[1]}, '
                                                f'{_run_state_vars[2]}, {_run_state_vars[3]}')
            
            return_val = None

            try:
                 return_val = self._beauty_instance.timeseries(request_time,
                                                                request_time,
                                                                _run_state_vars[0],
                                                                _run_state_vars[1],
                                                                _run_state_vars[2],
                                                                _run_state_vars[3],
                                                                regex=False,
                                                                all_publications=True)

            except ValueError as value_error:
                self._logger.info(f'Run Control status unavailable')

            if return_val is not None:
                
                if not isinstance(return_val, list):
                    raise ValueError(f'PBEAST request return {return_val} has unexpected '
                                            f'type {type(return_val)}. Expected type: list')

                print(len(return_val[0]))

                state = return_val[0][0]

                if state in target_states:

                    self._logger.info(f'Run Control status switched to {state}. '
                                                f'Continuing after delay of {return_delay}')
                    break

                else:
                    self._logger.info(f'Run Control status is {state}. Waiting '
                                        f'for state(s) {target_states} before continuing')

            request_duration = t.monotonic() - time_start

            if request_duration >= self._polling_interval.total_seconds():

                error_string = 'Request processing time '\
                                    'exceeded polling interval. '\
                                    f'Request processing time: {request_duration:.3f} s\t'\
                                    'Polling interval: '\
                                    f'{self._polling_interval.total_seconds()} s'

                self._logger.warning(error_string)

                # self._logger.error(error_string)
                # raise RuntimeError(error_string)

            else:
                delay_period = self._polling_interval.total_seconds() - request_duration
                await aio.sleep(delay_period)

        await aio.sleep(return_delay.total_seconds())

