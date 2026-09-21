#!/usr/bin/env node
import { createHash } from 'node:crypto';
import { readdirSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { spawnSync } from 'node:child_process';
import { isDeepStrictEqual } from 'node:util';

const expectedEngineVersion='twinleaf-adapter-0.1.0+c01ee150cbe47377';
const dataRoot=resolve(process.argv[2]??'data/competitive');
const reviewRoot=resolve(dataRoot,'teaching-reviews');
const reviews=readdirSync(reviewRoot).filter(name=>name.endsWith('.json')).sort()
  .map(name=>JSON.parse(readFileSync(resolve(reviewRoot,name),'utf8')))
  .filter(review=>review.reviewStatus==='reviewed'&&review.engineVersion===expectedEngineVersion);

if(reviews.length!==8)throw new Error(`Expected 8 reviewed snapshots pinned to ${expectedEngineVersion}; found ${reviews.length}.`);
const fixtureKeys=[...new Set(reviews.map(review=>`${review.familyId}\0${review.variationId}`))].sort();
const requests=fixtureKeys.map((key,index)=>{const [fixtureId,variationId]=key.split('\0');return{id:index+1,method:'fixture',params:{fixtureId,variationId}};});
const child=spawnSync(process.execPath,['packages/engine/dist/worker.cjs'],{
  input:requests.map(request=>JSON.stringify(request)).join('\n')+'\n',encoding:'utf8',maxBuffer:64*1024*1024
});
if(child.status!==0)throw new Error(child.stderr||`Worker exited ${child.status}`);
const responses=new Map(child.stdout.trim().split('\n').filter(Boolean).map(line=>{const value=JSON.parse(line);if(value.error)throw new Error(value.error.message);return[value.id,value.result];}));
const fixtures=new Map(requests.map(request=>[`${request.params.fixtureId}\0${request.params.variationId}`,responses.get(request.id)]));

const canonical=value=>{
  if(Array.isArray(value))return value.map(canonical);
  if(value&&typeof value==='object')return Object.fromEntries(Object.keys(value).sort().map(key=>[key,canonical(value[key])]));
  return value;
};
const pythonCompatibleJson=value=>JSON.stringify(canonical(value)).replace(/[\u007f-\uffff]/g,character=>`\\u${character.charCodeAt(0).toString(16).padStart(4,'0')}`);
const digest=value=>createHash('sha256').update(pythonCompatibleJson(value)).digest('hex');
const compact=value=>Object.fromEntries(Object.entries(value).filter(([,item])=>item!==undefined));
const binding=action=>compact({type:action.type,cardId:action.cardId,sourceRef:action.sourceRef,targetRef:action.targetRef});
const bindings=actions=>Object.fromEntries(actions.map(action=>[action.id,binding(action)]));
const differencePaths=(left,right,path='$',out=[])=>{
  if(isDeepStrictEqual(left,right)||out.length>=20)return out;
  if(!left||!right||typeof left!=='object'||typeof right!=='object'||Array.isArray(left)!==Array.isArray(right)){
    out.push(path);return out;
  }
  const keys=new Set([...Object.keys(left),...Object.keys(right)]);
  for(const key of keys)differencePaths(left[key],right[key],Array.isArray(left)?`${path}[${key}]`:`${path}.${key}`,out);
  return out;
};

const rows=reviews.map(review=>{
  const fixture=fixtures.get(`${review.familyId}\0${review.variationId}`);
  if(!fixture)throw new Error(`Missing reconstructed fixture for ${review.familyId}/${review.variationId}`);
  const savedLegal=bindings(review.observation.legalActions);
  const currentLegal=bindings(fixture.observation.legalActions);
  const acceptedSaved=Object.fromEntries(review.acceptableActionIds.map(id=>[id,savedLegal[id]]));
  const acceptedCurrent=Object.fromEntries(review.acceptableActionIds.map(id=>[id,currentLegal[id]]));
  return{
    reviewHash:review.reviewHash,
    teachingId:review.id,
    fixture:`${review.familyId}/${review.variationId}`,
    positionHash:review.positionHash,
    computedSavedPositionHash:digest(review.observation),
    computedCurrentPositionHash:digest(fixture.observation),
    observationEqual:isDeepStrictEqual(review.observation,fixture.observation),
    observationDifferences:differencePaths(review.observation,fixture.observation),
    legalBindingsEqual:isDeepStrictEqual(savedLegal,currentLegal),
    acceptedBindingsEqual:isDeepStrictEqual(acceptedSaved,acceptedCurrent),
  };
});
const passed=rows.every(row=>row.positionHash===row.computedSavedPositionHash&&row.positionHash===row.computedCurrentPositionHash&&row.observationEqual&&row.legalBindingsEqual&&row.acceptedBindingsEqual);
const report={
  schemaVersion:1,
  mode:'read-only-teaching-compatibility',
  sourceEngineVersion:expectedEngineVersion,
  reviewCount:rows.length,
  uniquePositionHashes:new Set(rows.map(row=>row.positionHash)).size,
  passed,
  rows,
};
process.stdout.write(JSON.stringify(report,null,2)+'\n');
if(!passed)process.exitCode=1;
