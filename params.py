event_ids_dict = {'EventMarker': {'DistractorPops': 1, 'TargetPops': 2, 'NoveltyPops': 3},
            'GazeRayIntersect': {'GazeRayIntersectsDistractor': 1, 'GazeRayIntersectsTarget': 2, 'GazeRayIntersectsNovelty': 3},
            'GazeBehavior': {'FixationDistractor': 6, 'FixationTarget': 7, 'FixationNovelty': 8, 'FixationNull': 9,
                              'Saccade2Distractor': 10, 'Saccade2Target': 11, 'Saccade2Novelty': 12,
                              'Saccade2Null': 13},
                  }  # event_ids_for_interested_epochs
# event_viz_groups = {'Fixation': ['FixationDistractor', 'FixationTarget', 'FixationNovelty', 'FixationNull'], 'Saccade': ['Saccade2Novelty', 'Saccade2Distractor', 'Saccade2Target', 'Saccade2Null']}
# event_viz_groups = {'Saccade2Novelty': 'Saccade2Novelty', 'Saccade2Distractor': 'Saccade2Distractor', 'Saccade2Target': 'Saccade2Target', 'Saccade2Null': 'Saccade2Null'}
event_viz = 'GazeRayIntersect'

event_color_dict = {1: 'b', 2: 'r', 3: 'orange', 4: 'black', 5: 'black', 6: 'b', 7: 'r', 8: 'orange', 9: 'grey', 10: 'orange'}
event_marker_color_dict = {}
