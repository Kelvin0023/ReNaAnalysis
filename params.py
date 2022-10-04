event_ids_dict = {'EventMarker': {'DistractorPops': 1, 'TargetPops': 2, 'NoveltyPops': 3},
            'GazeRayIntersect': {'GazeRayIntersectsDistractor': 4, 'GazeRayIntersectsTarget': 5, 'GazeRayIntersectsNovelty': 6},
            'GazeBehavior': {'FixationDistractor': 7, 'FixationTarget': 8, 'FixationNovelty': 9, 'FixationNull': 10,
                              'Saccade2Distractor': 11, 'Saccade2Target': 12, 'Saccade2Novelty': 13,
                              'Saccade2Null': 14},
                  }  # event_ids_for_interested_epochs
color_dict = {
              'DistractorPops': 'blue', 'TargetPops': 'red', 'NoveltyPops': 'orange',
              'Fixation': 'blue', 'Saccade': 'orange',
                "GazeRayIntersectsDistractor": 'blue', "GazeRayIntersectsTarget": 'red', "GazeRayIntersectsNovelty": 'orange',
              'FixationDistractor': 'blue', 'FixationTarget': 'red', 'FixationNovelty': 'orange', 'FixationNull': 'grey',
              'Saccade2Distractor': 'blue', 'Saccade2Target': 'red', 'Saccade2Novelty': 'orange', 'Saccade2Null': 'yellow'}

event_viz = 'GazeRayIntersect'
